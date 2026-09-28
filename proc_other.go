//go:build !unix

package main

import "os/exec"

func setProcessGroup(*exec.Cmd) {}
