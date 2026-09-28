package main

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/ironpark/acp-go/acp1"
)

// gitDir is never searched: it is large and never what the model wants.
const gitDir = ".git"

// walkFiles calls fn for every file under root, skipping gitDir and entries
// that cannot be read.
func walkFiles(ctx context.Context, root string, fn func(path string, d fs.DirEntry) error) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
		if d.IsDir() {
			if d.Name() == gitDir && path != root {
				return filepath.SkipDir
			}
			return nil
		}
		return fn(path, d)
	})
}

// ripgrepPath is where rg is installed, looked up once.
var ripgrepPath = sync.OnceValues(func() (string, error) { return exec.LookPath("rg") })

// globFiles finds files under root matching pattern, most recently modified
// first.
func globFiles(ctx context.Context, root, pattern string) (string, []acp1.ToolCallContent, error) {
	re, err := globRegexp(pattern)
	if err != nil {
		return "", nil, err
	}
	matchName := !strings.Contains(pattern, "/")

	type match struct {
		path    string
		modTime time.Time
	}
	var matches []match
	err = walkFiles(ctx, root, func(path string, d fs.DirEntry) error {
		rel, _ := filepath.Rel(root, path)
		subject := filepath.ToSlash(rel)
		if matchName {
			subject = d.Name()
		}
		if !re.MatchString(subject) {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		matches = append(matches, match{path, info.ModTime()})
		return nil
	})
	if err != nil {
		return "", nil, err
	}
	slices.SortFunc(matches, func(a, b match) int { return b.modTime.Compare(a.modTime) })

	if len(matches) == 0 {
		return "No files found.", []acp1.ToolCallContent{acp1.ToolText("No files found")}, nil
	}
	var b strings.Builder
	for _, m := range matches[:min(len(matches), globLimit)] {
		b.WriteString(m.path + "\n")
	}
	if len(matches) > globLimit {
		fmt.Fprintf(&b, "(Showing %d of %d matches. Use a more specific pattern.)", globLimit, len(matches))
	}
	return b.String(), []acp1.ToolCallContent{acp1.ToolText(fmt.Sprintf("Found %d files", len(matches)))}, nil
}

// globRegexp translates a glob into an anchored regexp: ** matches across
// directories, * and ? within one, [...] a character class and {a,b} any of
// the alternatives.
func globRegexp(pattern string) (*regexp.Regexp, error) {
	var b strings.Builder
	b.WriteString("^")
	depth := 0 // inside {...}
	for i := 0; i < len(pattern); i++ {
		c := pattern[i]
		switch c {
		case '*':
			if i+1 < len(pattern) && pattern[i+1] == '*' {
				i++
				if i+1 < len(pattern) && pattern[i+1] == '/' {
					i++
					b.WriteString("(?:.*/)?") // **/ matches zero or more directories
				} else {
					b.WriteString(".*")
				}
			} else {
				b.WriteString("[^/]*")
			}
		case '?':
			b.WriteString("[^/]")
		case '[':
			end := strings.IndexByte(pattern[i+1:], ']')
			if end < 0 {
				b.WriteString(`\[`)
				continue
			}
			class := pattern[i+1 : i+1+end]
			if strings.HasPrefix(class, "!") {
				class = "^" + class[1:]
			}
			b.WriteString("[" + class + "]")
			i += end + 1
		case '{':
			depth++
			b.WriteString("(?:")
		case '}':
			if depth > 0 {
				depth--
				b.WriteString(")")
			} else {
				b.WriteString(`\}`)
			}
		case ',':
			if depth > 0 {
				b.WriteString("|")
			} else {
				b.WriteString(",")
			}
		default:
			b.WriteString(regexp.QuoteMeta(string(c)))
		}
	}
	b.WriteString("$")
	re, err := regexp.Compile(b.String())
	if err != nil {
		return nil, fmt.Errorf("invalid glob %q: %w", pattern, err)
	}
	return re, nil
}

// grepFiles searches file contents under root, with ripgrep when it is
// installed and in Go otherwise.
func grepFiles(ctx context.Context, cwd, root, pattern, include string) (string, []acp1.ToolCallContent, error) {
	var (
		lines []string
		err   error
	)
	if rg, lookErr := ripgrepPath(); lookErr == nil {
		lines, err = ripgrep(ctx, rg, cwd, root, pattern, include)
	} else {
		lines, err = goGrep(ctx, root, pattern, include)
	}
	if err != nil {
		return "", nil, err
	}
	if len(lines) == 0 {
		return "No matches found.", []acp1.ToolCallContent{acp1.ToolText("No matches")}, nil
	}
	var b strings.Builder
	for _, line := range lines[:min(len(lines), grepLimit)] {
		b.WriteString(line + "\n")
	}
	summary := fmt.Sprintf("Found %d matches", len(lines))
	if len(lines) > grepLimit {
		fmt.Fprintf(&b, "(Showing the first %d matches. Narrow the pattern or path.)", grepLimit)
		summary = fmt.Sprintf("Found more than %d matches", grepLimit)
	}
	return b.String(), []acp1.ToolCallContent{acp1.ToolText(summary)}, nil
}

func ripgrep(ctx context.Context, rg, cwd, root, pattern, include string) ([]string, error) {
	args := []string{"--line-number", "--no-heading", "--color=never", "--hidden", "--max-columns=500", "--glob=!" + gitDir}
	if include != "" {
		args = append(args, "--glob="+include)
	}
	args = append(args, "--regexp", pattern, "--", root)
	cmd := exec.CommandContext(ctx, rg, args...)
	cmd.Dir = cwd
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	var lines []string
	scanner := bufio.NewScanner(out)
	scanner.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
		if len(lines) > grepLimit {
			break // one past the limit tells that there are more
		}
	}
	if len(lines) > grepLimit {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return lines, nil
	}
	if err := cmd.Wait(); err != nil {
		// ripgrep exits 1 when nothing matched.
		if exitErr, ok := errors.AsType[*exec.ExitError](err); ok && exitErr.ExitCode() == 1 {
			return lines, nil
		}
		if msg := strings.TrimSpace(stderr.String()); msg != "" {
			return nil, errors.New(msg)
		}
		return nil, err
	}
	return lines, nil
}

func goGrep(ctx context.Context, root, pattern, include string) ([]string, error) {
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid regular expression: %w", err)
	}
	var includeRe *regexp.Regexp
	if include != "" {
		if includeRe, err = globRegexp(include); err != nil {
			return nil, err
		}
	}
	var lines []string
	errEnough := errors.New("enough matches")
	err = walkFiles(ctx, root, func(path string, d fs.DirEntry) error {
		if includeRe != nil && !includeRe.MatchString(d.Name()) {
			return nil
		}
		matches, err := grepFile(path, re, grepLimit+1-len(lines))
		if err != nil {
			return nil // unreadable
		}
		lines = append(lines, matches...)
		if len(lines) > grepLimit {
			return errEnough
		}
		return nil
	})
	if err != nil && !errors.Is(err, errEnough) {
		return nil, err
	}
	return lines, nil
}

// grepFile returns up to limit matching lines of a file as path:line:text,
// reading it line by line. Binary files have no matches.
func grepFile(path string, re *regexp.Regexp, limit int) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 64<<10)
	if head, _ := r.Peek(8 << 10); bytes.IndexByte(head, 0) >= 0 {
		return nil, nil
	}
	var matches []string
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64<<10), 4<<20)
	for n := 1; scanner.Scan() && len(matches) < limit; n++ {
		line := scanner.Bytes()
		if !re.Match(line) {
			continue
		}
		if len(line) > 500 {
			line = append(line[:500:500], "…"...)
		}
		matches = append(matches, fmt.Sprintf("%s:%d:%s", path, n, line))
	}
	return matches, nil
}
