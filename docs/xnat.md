# XNAT Support

This package now includes support for XNAT servers in addition to OMERO servers.

## Installation

To use XNAT features, install with the xnat extra:

```bash
pip install lavlab-python-utils[xnat]
```

## Configuration

Configure XNAT service in your configuration file (`~/.lavlab.yml` or `/etc/lavlab.yml`):

```yaml
histology:
  service:
    name: 'xnat'
    host: 'https://your-xnat-server.org'
    username: your_username  # optional, will prompt if not provided
    passwd: your_password    # optional, recommended to use keyring instead
```

## Usage

### Basic Connection

```python
import lavlab.xnat

# Connect using configuration
session = lavlab.xnat.connect()

# Or create service provider directly
provider = lavlab.xnat.XNATServiceProvider()
session = provider.login()
```

### Helper Functions

```python
from lavlab.xnat.helpers import get_projects, get_subjects, get_experiments

# Get available projects
projects = get_projects(session)

# Get subjects for a project
subjects = get_subjects(session, "PROJECT_ID")

# Get experiments for a subject
experiments = get_experiments(session, "PROJECT_ID", "SUBJECT_ID")

# Download scan files
from lavlab.xnat.helpers import download_scan_file

with download_scan_file(session, "EXPERIMENT_ID", "SCAN_ID", "filename.dcm") as file_path:
    # Work with downloaded file
    pass
# File is automatically cleaned up
```

### Search and Discovery

```python
from lavlab.xnat.helpers import find_experiments_by_type, search_experiments

# Find MR experiments in a project
mr_experiments = find_experiments_by_type(session, "PROJECT_ID", "xnat:mrSessionData")

# Search experiments with custom criteria
results = search_experiments(session, project="PROJECT_ID", modality="MR")
```

## Service Provider Architecture

The XNAT support follows the same service provider pattern as OMERO:

- `XNATServiceProvider` handles authentication and connection
- Credentials are managed through keyring for security
- The service is registered as an entry point for dynamic loading
- Configuration follows the same patterns as other services