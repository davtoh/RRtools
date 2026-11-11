# -*- coding: utf-8 -*-
"""
Logging utilities, stdout control, redirection, and multiplexing contexts.
"""
import sys
import io

__all__ = [
    "StdoutSIM",
    "StdoutMULTI",
    "StdoutLOG",
    "Controlstdout",
    "stdout",
]


class StdoutSIM:
    """
    Simple logger to simulate stdout output adding a closed control.
    """

    def __init__(self, stdout=None, closed=False):
        """
        :param stdout: file-like object to write
        :param closed: If True disables output. Else prints normally.
        """
        if stdout is None:
            stdout = sys.stdout
        self.stdout = stdout
        self.closed = closed

    def write(self, text, **kwargs):
        if not self.closed:
            self.stdout.write(text, **kwargs)

    def printline(self, text, **kwargs):
        if not self.closed:
            self.stdout.printline(text, **kwargs)

    def printlines(self, lines, **kwargs):
        if not self.closed:
            self.stdout.printlines(lines, **kwargs)

    def flush(self):
        if not self.closed:
            self.stdout.flush()

    def close(self):
        """
        Do nothing.
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


stdout = StdoutSIM()


class StdoutMULTI:
    """
    Enclose several file-like objects.

    :param file_list: list of file-like objects
    """

    def __init__(self, file_list):
        self.file_list = file_list

    def write(self, text, **kwargs):
        for log in self.file_list:
            log.write(text, **kwargs)

    def printline(self, text, **kwargs):
        for log in self.file_list:
            log.printline(text, **kwargs)

    def printlines(self, lines, **kwargs):
        for log in self.file_list:
            log.printlines(lines, **kwargs)

    def flush(self, **kwargs):
        for log in self.file_list:
            log.flush(**kwargs)

    def close(self, **kwargs):
        for log in self.file_list:
            log.close(**kwargs)

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class StdoutLOG(StdoutMULTI):
    """
    Simple logger to save stdout output so anything printed in the console
    is logged to a file.

    :param path: path to logging file
    :param mode: mode for opening the file.
    :param add_stdout: if True closes previous logs and continues with new log
    """

    def __init__(self, path, mode="w+", add_stdout=True):
        self.closed = False
        self._old = sys.stdout  # register old logger
        self._cleaned = False

        if hasattr(path, "write"):  # must be file-like
            self._file = path
            self._path = None
            self.mode = None
        else:
            self._file = open(path, mode)
            self._path = path
            self.mode = mode

        # keep logs to write to
        if add_stdout:
            file_list = [self._old, self._file]
        else:
            file_list = [self._file]
        super().__init__(file_list=file_list)
        sys.stdout = self  # register log

    def close(self, **kwargs):
        """
        Close StdoutLOG but does not ensure cleanup to restore sys.stdout.
        """
        if not self.closed:
            # close file
            self._file.close()

            # unregister this log
            if self._file in self.file_list:
                # removes from file_list of loggers to prevent output
                # but sys.stdout is not replaced with old to not affect
                # anything done after the instance creating
                self.file_list.remove(self._file)

            self.closed = True
            self._cleanup()

    def _cleanup(self):
        """
        Ensure sys.stdout is restored after all StdoutLOG are closed.
        """
        if not self._cleaned and self.closed:
            # if possible restore sys.stdout as it was before
            if sys.stdout is self:  # output not replaced
                sys.stdout = self._old  # return old log

                # at least this node has been cleaned
                self._cleaned = True

                try:  # try to clean up old references
                    self._old._cleanup()
                    # overheats more but deletes circular references
                    if self._old._cleaned and self._old in self.file_list:
                        self.file_list.remove(self._old)
                except AttributeError:
                    pass  # old node not cleaned


class Controlstdout:
    """
    Context manager to control output to stdout.

    :param disable: if True suppress output.
    :param buffer: (None) if True creates a buffer to collect all data
            printed to stdout which can be retrieved with self.buffered.
            A file can be given but if it is write-only it cannot retrieve
            data to self.buffered so "w+" is recommended.

    .. warning:: If a reference to sys.stdout is kept before the Controlstdout
            instance then output can be printed through it and cannot be
            controlled by the Controlstdout context.
    """

    def __init__(self, disable=True, buffer=None):
        self.disable = disable
        self.buffer = buffer
        self.buffered = ""
        self.stdout_old = None
        self.stdout_new = None

    def __enter__(self):
        self.stdout_old = sys.stdout
        self.stdout_new = StdoutSIM(closed=self.disable)
        if self.buffer:
            if self.buffer is True:
                self.buffer = io.StringIO()
            self.stdout_new = StdoutMULTI([self.stdout_new, self.buffer])
        sys.stdout = self.stdout_new
        return self

    def __exit__(self, type, value, traceback):
        if self.buffer:
            try:
                self.buffer.seek(0)
                self.buffered = self.buffer.read()
                self.buffer.close()
            except Exception:
                pass
        sys.stdout = self.stdout_old

