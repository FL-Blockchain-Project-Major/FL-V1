import pytest

from client1 import Client1


def test_client_exits_after_first_fit():
    client = Client1.__new__(Client1)
    client._training_completed = False

    with pytest.raises(SystemExit):
        client._shutdown_after_first_fit()

    assert client._training_completed is True

    client._shutdown_after_first_fit()
    assert client._training_completed is True
