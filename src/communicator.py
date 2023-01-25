import pickle
import struct
import socket
import logging
import sys

from typing import Tuple


logger = logging.getLogger(__name__)
logger.propagate = False
handler_console = logging.StreamHandler(stream=sys.stdout)
format_console = logging.Formatter('%(asctime)s [%(levelname)s]: %(name)s : %(message)s')
handler_console.setFormatter(format_console)
handler_console.setLevel(logging.DEBUG)
logger.addHandler(handler_console)


class Communicator(object):


    def __init__(self, sock=None, ip_address=None):
        self.ip = ip_address
        self.sock = socket.socket() if sock == None else sock

    def send_msg(self, msg):
        msg_pickle = pickle.dumps(msg)
        self.sock.sendall(struct.pack(">I", len(msg_pickle)))
        self.sock.sendall(msg_pickle)
        logger.debug(
            msg[0],
            f'sent to {self.sock.getpeername()[0]}:', 
            self.sock.getpeername()[1]
        )

    def recv_msg(self, expect_msg_type=None):
        msg_len = struct.unpack(">I", self.sock.recv(4))[0]
        msg = self.sock.recv(msg_len, socket.MSG_WAITALL)
        msg = pickle.loads(msg)
        logger.debug(
            msg[0],
            f'received from {self.sock.getpeername()[0]}:', 
            self.sock.getpeername()[1]
        )
        if expect_msg_type is not None:
            if msg[0] == 'Finish':
                return msg
            elif msg[0] != expect_msg_type:
                raise Exception("Expected " + expect_msg_type + " but received " + msg[0])
        return msg

    def connect(self, conn_tuple: Tuple[str, int]): 
        self.sock.connect(conn_tuple)
