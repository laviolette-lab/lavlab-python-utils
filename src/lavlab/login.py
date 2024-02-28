import getpass
import keyring
import logging
LOGGER = logging.getLogger(__name__)
def keyring_get(service:str, property:str):
    """
    Gets username and password from keyring.
    """
    val = keyring.get_password('lavlab-python-utils.'+service, property)
    return val
def keyring_set(service:str, property:str, value:str):
    """
    Gets username and password from keyring.
    """
    val = keyring.set_password('lavlab-python-utils.'+service, property, value)
    return val

def get_username_password(service:str):
    """
    Gets username and password from keyring.
    """
    username = keyring_get(service, 'username')
    password = keyring_get(service, username)
    if username is None or password is None:
        return None, None
    LOGGER.info(f"Found username and password in keyring for service: {service}")
    return username, password

def prompt_username_password(service:str):
    """
    Prompts user for username and password and asks if they want to save them to keyring.
    """
    username = input('Username: ')
    password = getpass.getpass('Password: ')
    return username, password

def prompt_kwargs():
    """
    Prompts user for server information. Yields kwargs
    """
    while True:
        key = input('key: ')
        if key is None or key == '':
            break
        print(f'key: {key}\nPlease enter a value for this key.')
        val = input('value: ')
        if val is None or val == '':
            break
        yield key, val

