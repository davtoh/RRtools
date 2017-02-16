#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
from builtins import filter
from builtins import range
from errno import ECONNREFUSED
from functools import partial
from multiprocessing import Pool
import socket
# http://codereview.stackexchange.com/questions/38452/python-port-scanner
NUM_CORES = 4


def ping(host, port):
    try:
        socket.socket().connect((host, port))
        print(str(port) + " Open")
        return port
    except socket.error as err:
        if err.errno == ECONNREFUSED:
            return False
        raise


def scan_ports(host):
    p = Pool(NUM_CORES)
    ping_host = partial(ping, host)
    return list(filter(bool, p.map(ping_host, list(range(1, 65536)))))


def main(host=None):
    if host is None:
        host = "127.0.0.1"

    print("\nScanning ports on " + host + " ...")
    ports = list(scan_ports(host))
    print("\nDone.")

    print(str(len(ports)) + " ports available.")
    print(ports)


if __name__ == "__main__":
    main()