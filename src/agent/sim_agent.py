import orjson
import socket
import sys
from typing import Optional

from logger import logger
from agent.sim_config import create_config


class SimRequest:
    def __init__(self, command, body):
        self._command = command
        self._body = body

    def serialize(self):
        return orjson.dumps(
            {
                "command": self._command,
                "body": self._body,
            }
        )


class StartSimSession(SimRequest):
    def __init__(self, sim_config):
        super().__init__("START_SIM_SESSION", {"RaidSimRequest": sim_config})


class GetState(SimRequest):
    def __init__(self):
        super().__init__("GET_STATE", None)


class Cast(SimRequest):
    def __init__(self, spell):
        super().__init__("CAST", {"spell": spell})


class WaitDuration(SimRequest):
    def __init__(self, duration):
        super().__init__("WAIT_DURATION", {"duration": duration})


class SimResponse:
    def __init__(self, raw_response):
        self._raw_response = raw_response
        self._json = orjson.loads(raw_response)

    @property
    def success(self):
        return self._json["Success"]

    @property
    def body(self):
        return self._json["Body"]


class SimConnection:
    def __init__(self, port):
        self._connection: Optional[socket.socket] = None
        self._port = port

    def connect(self):
        self._connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._connection.settimeout(1.0)
        self._connection.connect(self._port)

    def disconnect(self):
        try:
            self._connection.close()
        except:
            pass

        self._connection = None

    @property
    def is_connected(self):
        return self._connection is not None

    def send_request(self, request: SimRequest):
        if not self.is_connected:
            raise Exception("Not connected")

        body = request.serialize()
        logger.debug("Sending request", body)
        payload = len(body).to_bytes(4, byteorder="little") + body
        self._connection.sendall(payload)
        logger.debug("finished sending request")

        buffer = self._connection.recv(1024*15)
        response_length = int.from_bytes(buffer[:4], byteorder="little")
        response = buffer[4:4 + response_length]

        response = SimResponse(response)
        assert response.success
        return response.body

    # pickle support
    def __getstate__(self):
        return self._port

    def __setstate__(self, port):
        self._port = port
        self._connection = None


class SimAgent:
    def __init__(self, port):
        self._connection = SimConnection(port)
        self._state = None

    def close(self):
        self._connection.disconnect()

    def reset(self, sim_config):
        if self._connection.is_connected:
            self._connection.send_request(StartSimSession(sim_config))
        else:
            self._connection.connect()
            self._connection.send_request(StartSimSession(sim_config))

        self._state = None
        return self._refetch_state()

    def cast(self, spell):
        response = self._connection.send_request(Cast(spell))
        self._step()
        return response

    def wait(self, duration):
        response = self._connection.send_request(WaitDuration(duration))
        self._refetch_state()
        return response

    def _step(self):
        return self.wait(50)

    def do_nothing(self):
        return self._step()

    def _refetch_state(self):
        self._state = self._connection.send_request(GetState())
        return self._state

    def get_state(self):
        if self._state is not None:
            return self._state
        return self._refetch_state()


if __name__ == "__main__":
    agent = SimAgent(port=sys.argv[1])

    sim_config = create_config(random_seed=0)
    state = agent.reset(sim_config)
    print(state)
