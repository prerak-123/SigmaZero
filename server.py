import socket
import time
import utils
import json
import config
import numpy as np
import sys
import threading
import utils
import multiprocessing
import logging
from typing import Tuple


logging.basicConfig(level=logging.INFO, format=' %(message)s')

class ServerSocket:
	def __init__(self, host, port):
		"""
		The server object listens to connections and creates client handlers
		for every client (multi-threaded).

		It receives inputs from the clients and returns the predictions to the correct client.
		"""
		self.host = host
		self.port = port


	def start(self):
		"""
		Start the server and listen for connections.
		"""
		logging.info(f"Starting server on {self.host}:{self.port}...")
		self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
		self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
		self.sock.bind((self.host, self.port))
		# listen for incoming connections, queue up to 24 requests
		self.sock.listen(24)
		logging.info(f"Server started on {self.sock.getsockname()}")
		try:
			while True:
				self.accept()
				logging.info(f"Current thread count: {threading.active_count()}.")
		except KeyboardInterrupt:
			self.stop()
		except Exception as e:
			logging.debug(f"Error: {e}")
			self.sock.close()

	def accept(self):
		"""
		Accept a connection and create a client handler for it.	
		"""
		logging.info("Waiting for client...")
		self.client, address = self.sock.accept()
		logging.info(f"Client connected from {address}")
		clh = ClientHandler(self.client, address)
		# start new thread to handle client
		clh.start()

	def stop(self):
		logging.info("Stopping server...")
		self.sock.close()
		logging.info("Server stopped.")


class ClientHandler(threading.Thread):
	def __init__(self, sock: socket.socket, address: Tuple[str, int]):
		"""
		The ClientHandler object handles a single client connection, and sends
		inputs to the server, and returns the server's predictions to the client.
		"""
		super().__init__()
		self.BUFFER_SIZE = config.SOCKET_BUFFER_SIZE
		self.sock = sock
		self.address = address

	def run(self):
		"""Create a new thread"""
		print(f"ClientHandler started.")
		while True:
			data = self.receive()
			if data is None or len(data) == 0:
				self.close()
				break
			data = np.array(np.frombuffer(data, dtype=bool))
			data = data.reshape(1, *config.INPUT_SHAPE)
			data = tf.convert_to_tensor(data, dtype=tf.bool)
			# make prediction
			p, v = predict(data)
			p, v = p[0].numpy().tolist(), float(v[0][0])
			response = json.dumps({"prediction": p, "value": v})
			self.send(f"{len(response):010d}".encode('ascii'))
			self.send(response.encode('ascii'))

	def receive(self):
		"""
		Receive data from the client.
		"""
		data = None
		try:
			data_length = self.sock.recv(10)
			if data_length == b'':
				# this happens if the socket connects and then closes without sending data
				return data
			data_length = int(data_length.decode("ascii"))
			data = utils.recvall(self.sock, data_length)
			if len(data) != 1216:
				data = None
				raise ValueError("Invalid data length, closing socket")
		except ConnectionResetError:
			logging.warning(f"Connection reset by peer. Client IP: {str(self.address[0])}:{str(self.address[1])}")
		except ValueError as e:
			logging.warning(e)
		return data

	def send(self, data):
		"""
		Send data to the client.
		"""
		self.sock.send(data)

	def close(self):
		"""
		Close the client connection.
		"""
		self.sock.close()