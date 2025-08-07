"""General helper functions for writing XNAT utilities."""

from contextlib import contextmanager
from typing import Union, List

import xnat  # type: ignore

import lavlab.xnat

LOGGER = lavlab.xnat.LOGGER.getChild("helpers")


## PROJECT/SUBJECT/SESSION CONTEXT HELPERS


def get_projects(session: xnat.XNATSession) -> List[str]:
    """Get list of available project IDs.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.

    Returns
    -------
    List[str]
        List of project IDs.
    """
    return list(session.projects.keys())


def get_subjects(session: xnat.XNATSession, project_id: str) -> List[str]:
    """Get list of subject IDs for a given project.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    project_id : str
        Project ID.

    Returns
    -------
    List[str]
        List of subject IDs.
    """
    project = session.projects[project_id]
    return list(project.subjects.keys())


def get_experiments(session: xnat.XNATSession, project_id: str, subject_id: str) -> List[str]:
    """Get list of experiment IDs for a given project and subject.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    project_id : str
        Project ID.
    subject_id : str
        Subject ID.

    Returns
    -------
    List[str]
        List of experiment IDs.
    """
    project = session.projects[project_id]
    subject = project.subjects[subject_id]
    return list(subject.experiments.keys())


## DATA ACCESS HELPERS


def get_scan_files(session: xnat.XNATSession, experiment_id: str, scan_id: str) -> List:
    """Get files associated with a scan.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    experiment_id : str
        Experiment ID.
    scan_id : str
        Scan ID.

    Returns
    -------
    List
        List of file objects.
    """
    experiment = session.experiments[experiment_id]
    scan = experiment.scans[scan_id]
    return list(scan.files.values())


@contextmanager
def download_scan_file(session: xnat.XNATSession, experiment_id: str, scan_id: str, filename: str):
    """Context manager to download and automatically clean up a scan file.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    experiment_id : str
        Experiment ID.
    scan_id : str
        Scan ID.
    filename : str
        Name of the file to download.

    Yields
    ------
    str
        Path to the downloaded file.
    """
    experiment = session.experiments[experiment_id]
    scan = experiment.scans[scan_id]
    
    try:
        file_path = scan.files[filename].download()
        yield file_path
    finally:
        # Clean up downloaded file if needed
        import os
        if os.path.exists(file_path):
            os.remove(file_path)


## SEARCH HELPERS


def find_experiments_by_type(session: xnat.XNATSession, project_id: str, experiment_type: str) -> List:
    """Find experiments of a specific type within a project.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    project_id : str
        Project ID to search within.
    experiment_type : str
        Type of experiment to search for (e.g., 'xnat:mrSessionData').

    Returns
    -------
    List
        List of matching experiment objects.
    """
    project = session.projects[project_id]
    matching_experiments = []
    
    for experiment in project.experiments.values():
        if experiment.attrs.get('xsi:type') == experiment_type:
            matching_experiments.append(experiment)
    
    return matching_experiments


def search_experiments(session: xnat.XNATSession, **search_criteria) -> List:
    """Search for experiments based on criteria.

    Parameters
    ----------
    session : xnat.XNATSession
        XNAT session object.
    **search_criteria
        Search criteria as keyword arguments.

    Returns
    -------
    List
        List of matching experiment objects.
    """
    # This is a basic implementation - could be extended with more sophisticated search
    table = session.experiments.tabulate(**search_criteria)
    return table