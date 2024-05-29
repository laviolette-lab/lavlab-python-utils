# Implementing the Service Provider

## Overview

This guide covers how to implement a new service provider in LavLab Python Utils.

## Steps
Adding a new service provider is often a smaller step in adding a new service as to the toolkit, you'll likely need to add it as a [new feature module](new_feature.md).
1. **Identify the service**: Determine the service that needs to be provided.
2. **Implement the service**: Create a new class that implements the service interface.
3. **Register the service**: Add the new service implementation to the service registry.

## Example
1. **Identify the service**:

We'll use the omero implementation as an example, we'll need to use the omero-py's BlitzGateway.

2. **Implement the service**:
```python
# lavlab/omero/__init__.py
from lavlab.login import AbstractServiceProvider

class OmeroServiceProvider(AbstractServiceProvider):  # pylint: disable=R0903
    """
    Provides a connection to a defined OMERO server using omero-py.
    """

    SERVICE = "OMERO"

    def login(self) -> BlitzGateway:
        """
        Logins into configured omero server.

        Returns
        -------
        BlitzGateway
            OMERO API gateway

        Raises
        ------
        RuntimeError
            Could not login to OMERO server.
        """
        details = lavlab.ctx.histology.service.copy()
        if details.get("username") is None or details.get("passwd") is None:
            username, password = self.cred_provider.get_credentials()
            details.update({"username": username, "passwd": password})

        conn = BlitzGateway(**details)
        if conn.connect():
            return conn
        raise RuntimeError("Unable to connect to OMERO server.")

```
Let's break it down.
1. Create a subclass from the AbstractServiceProvider (and document it):
```python
class OmeroServiceProvider(AbstractServiceProvider):
    """
    Provides a connection to a defined OMERO server using omero-py.
    """
```
2. We need to name the service we're providing like so:
```python
# this allows us to create a service provider from our configuration
SERVICE = "OMERO"
```
3. (OPTIONAL) Configure the credential provider:
```python
CREDENTIAL_PROVIDER = YourCredentialProvider
```
The abstract implementation uses the KeychainCredentialsProvider, this uses the configs or the systems keychain to access a username and password. **For Basic login flows you do not need to configure a credential provider.** Should your service use OAuth2 or some mechanism not based on a Basic login flow, you may need to implement a CrendentialProvider.

4. Implement the login function:
```python
def login(self) -> BlitzGateway:
    """
    Logins into configured omero server.

    Returns
    -------
    BlitzGateway
        OMERO API gateway

    Raises
    ------
    RuntimeError
        Could not login to OMERO server.
    """
    details = lavlab.ctx.histology.service.copy()
    if details.get("username") is None or details.get("passwd") is None:
        username, password = self.cred_provider.get_credentials()
        details.update({"username": username, "passwd": password})

    conn = BlitzGateway(**details)
    if conn.connect():
        return conn
    raise RuntimeError("Unable to connect to OMERO server.")
```
There's typically not much work to be done in the login class. Most of the work is expected to be done by the service, with login settings being configured in our lavlab.yml file:
```yaml
# be sure to match the service with the datatype provided
histology:
  service:
    name: 'omero' 
    # BlitzGateway kwargs
    host: null
    username: null 
    passwd: null
    ...
```

3. **Register the service**:

Super simple, just gotta add our new service provider to our pyproject.toml.

```toml
[project.entry-points."lavlab_python_utils.service_providers"]
OMERO = "lavlab.omero:OmeroServiceProvider"
```

## Testing

Ensure that you write tests for your new service provider and run them to verify that everything works correctly. Service providers are a bit tough to write tests for, as you need a server to test against. Currently for OMERO we run it against a server in our local deployment using self hosted runner. Ideally you'd get some sort of public test server to run it against. See [test/omero](https://github.com/laviolette-lab/lavlab-python-utils/tree/main/test/omero) for more how we test our OMERO service provider.