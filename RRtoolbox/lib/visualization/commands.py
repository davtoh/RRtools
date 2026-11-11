# -*- coding: utf-8 -*-
"""
Command processor and keystroke interpreter mixin for Plotim.
"""

import sys
import traceback
from .helpers import limitaxis

__all__ = ["CommandMixin"]


class CommandMixin(object):
    """Mixin providing command parsing, history, and keystroke bindings for Plotim."""

    def _eval_single_command(self, command, showresult=True, wait=3):
        """Evaluate a single command string on self."""
        try:
            if "=" in command or "(" in command or "." in command:
                result = locals()
                exec("_ = {}".format(command), result)
                if showresult:
                    self.plotintime(items=[[command + " executed"]], wait=wait)
                    if result.get("_"):
                        self.plotintime(items=[[result["_"]]], wait=wait)
            else:
                obj = getattr(self, command)
                if isinstance(obj, bool):
                    setattr(self, command, not obj)
                    if showresult:
                        status = "is ON" if getattr(self, command) else "is OFF"
                        self.plotintime(items=[["{} {}".format(command, status)]], wait=wait)
                elif callable(obj):
                    result = obj()
                    if showresult:
                        self.plotintime(items=[["{}".format(result)]], wait=wait)
            return True
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
            msg = lines[-1]
            self.plotintime(items=[["Error executing " + command + ": " + msg]],
                            wait=wait, bgrcolor=self.errorbackground)
            print(msg)
            return False

    def _eval_command_sequence(self, cmd, showresult=True, wait=3):
        """Evaluate string or list of commands with error reporting."""
        commands = cmd if isinstance(cmd, list) else [cmd]
        for command in commands:
            if not self._eval_single_command(command, showresult, wait):
                return False
        if showresult:
            self.plotintime(items=[[self.cmd + " executed"]], wait=wait)
        return True

    def cmdfunc(self, execute=False):
        """Command processor and decoupled solver for Plotim."""
        text = []
        if execute and self.cmd != "":
            choice = [x for x in self.cmdlist if x.startswith(self.cmd)]
            if choice:
                self.cmd = choice[0]
            command = self.cmdeval.get(self.cmd)
            if command is None:
                command = self.formatcmd(self.cmd) if self.cmdformatter else self.cmd
            if self._eval_command_sequence(command):
                command2 = self.cmdeval.get("end")
                if command2 is not None:
                    self._eval_command_sequence(command2, False)
                self.cmderror = False
                if not self.cmdcache.count(self.cmd):
                    self.cmdcache.append(command)
                self.cmd = ""
            else:
                self.cmderror = True
        else:
            text.extend([["cmd: " + self.cmd]])

        self._cmdbgrcolor = self.errorbackground if (self.cmderror and self.cmd != "") else self.textbackground
        self._cmditmes = text

    def formatcmd(self, cmd, references=("+", "-", "*", "="), lmissing="self."):
        """Format commands by prefixing missing self references."""
        def splitter(c, refs):
            for i in refs:
                c = c.replace(i, " ")
            return c.split()

        def correct(string, missing):
            try:
                eval(string)
            except Exception:
                if lmissing not in string:
                    string = missing + string
            return string

        strlist = splitter(cmd, references)
        if len(strlist) != 1:
            for s in strlist:
                cmd = cmd.replace(s, correct(s, lmissing))
        return cmd.replace(" ", "")

    def _handle_arrow_keys(self, key):
        """Handle up/down/left/right history and filter navigation."""
        if key in (2490368, 2621440):  # Up / Down
            if not self.cmdfiltered:
                self.cmdfiltered = self.cmdlist if self.cmd == "" else [x for x in self.cmdlist if x.startswith(self.cmd)]
            mylist = self.cmdfiltered
            if mylist:
                delta = -1 if key == 2490368 else 1
                self.cmdfiltered_choice = limitaxis(self.cmdfiltered_choice + delta, len(mylist) - 1)
                self.cmd = mylist[self.cmdfiltered_choice]
                toshow = [["* " + mylist[i]] if i == self.cmdfiltered_choice else [mylist[i]] for i in range(len(mylist))]
                self._cmditmes = [["cmd: " + self.cmd]] + toshow
            return True
        if key in (2555904, 2424832):  # Right / Left (cache)
            mylist = self.cmdcache
            if mylist:
                delta = 1 if key == 2555904 else -1
                self.cmdcache_choice = limitaxis(self.cmdcache_choice + delta, len(mylist) - 1)
                self.cmd = mylist[self.cmdcache_choice]
                toshow = [["* " + mylist[i]] if i == self.cmdcache_choice else [mylist[i]] for i in range(len(mylist))]
                self._cmditmes = [["cmd: " + self.cmd]] + toshow
            return True
        return False

    def _handle_clipboard_and_buffer(self, key):
        """Handle Ctrl+C, Ctrl+V, Ctrl+X, Ctrl+Z and Backspace."""
        if key == 8:  # Backspace
            self.cmd = self.cmd[:-1]
        elif key == 22:  # Ctrl+V
            if self.cmdbuffer == [] or self.cmd != self.cmdbuffer[-1]:
                self.cmdbuffer.append(self.cmd)
            self.cmd += self.clipboard.paste()
        elif key == 3:  # Ctrl+C
            self.clipboard.copy(self.cmd)
        elif key == 24:  # Ctrl+X
            if self.cmdbuffer == [] or self.cmd != self.cmdbuffer[-1]:
                self.cmdbuffer.append(self.cmd)
            self.clipboard.copy(self.cmd)
            self.cmd = ""
        elif key == 26:  # Ctrl+Z
            self.cmd = self.cmd[:-1] if not self.cmdbuffer else self.cmdbuffer.pop()
        elif key not in (13, 0, -1) and key < 255:
            try:
                self.cmd += chr(key)
            except Exception:
                pass

    def builtincmd(self):
        """Internal cmd keystroke control."""
        self.iscmd = False
        key = self.pressedkey
        if key is None or key in (255, 27):
            return self.iscmd

        if self._handle_arrow_keys(key):
            self.iscmd = True
            return True

        if key == 13:  # Enter
            if self.cmdbuffer == [] or self.cmd != self.cmdbuffer[-1]:
                self.cmdbuffer.append(self.cmd)
            self.cmdfunc(True)
            self.iscmd = True
            return self.iscmd

        self._handle_clipboard_and_buffer(key)
        if self.cmd != "":
            mylist = [x for x in self.cmdlist if x.startswith(self.cmd)]
            self._cmditmes = [["cmd: " + self.cmd]] + ([[i] for i in mylist] if mylist else [])
            self.iscmd = True
        else:
            self.iscmd = self._cmditmes is not None
            self._cmditmes = None

        return self.iscmd

