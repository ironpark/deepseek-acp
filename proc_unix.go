//go:build unix

package main

import (
	"os/exec"
	"syscall"
)

// setProcessGroup runs the command in its own process group and kills the
// whole group when the command is cancelled, so no child outlives it.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Cancel = func() error {
		return syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
	}
}
