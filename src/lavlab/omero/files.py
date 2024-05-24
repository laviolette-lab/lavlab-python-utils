"""OMERO File utiltities"""

import os
from typing import Optional

from omero.gateway import (  # type: ignore
    BlitzGateway,
    FileAnnotationWrapper,
    ImageWrapper,
)

import lavlab.omero

LOGGER = lavlab.omero.LOGGER.getChild("files")


def download_file_annotation(file_annot: FileAnnotationWrapper, outdir=".") -> str:
    """
    Downloads FileAnnotation from OMERO into a local directory.

    Parameters
    ----------
    file_annot: omero.gateway.FileAnnotationWrapper
        Remote Omero File Annotation object.
    out_dir: str, Default: '.'
        Where to download this file.

    Returns
    -------
    str
        String path to downloaded file.
    """
    path = os.path.abspath(outdir) + os.sep + file_annot.getFile().getName()
    print(f"Downloading {path}...")
    with open(path, "wb") as f:
        for chunk in file_annot.getFileInChunks():
            f.write(chunk)
    print(f"{path} downloaded!")
    return path


def get_script_by_name(
    conn: BlitzGateway, fn: str, absolute=False, check_user_scripts=False
) -> int:
    """
    Searches for an omero script in the host with the given name.

    Parameters
    ----------
    conn: omero.gateway.BlitzGateway
        An Omero BlitzGateway with a session.
    fn: str
        Name of remote Omero.Script
    absolute: bool, Default: False
        Absolute uses getScriptID(). This method does not accept wildcards and requires a path.
        Default will get all remote script names and compare the remote filename to fn.
    checkUserScripts: bool, Default: False
        Not implemented.

    Returns
    -------
    int
        Omero.Script Id
    """
    if check_user_scripts:
        LOGGER.warning(
            "getScriptByName not fully implemented! May cause unexpected results!"
        )
    script_svc = conn.getScriptService()
    try:
        if absolute is True:
            return script_svc.getScriptID(fn)
        for script in script_svc.getScripts():
            if script.getName().getValue() == fn:
                return script.getId().getValue()
        raise ValueError("Could not find script!")
    finally:
        script_svc.close()


def upload_file_as_annotation(
    parent_obj: ImageWrapper,
    file_path: str,
    namespace: str,
    mime: Optional[str] = None,
    overwrite=True,
) -> FileAnnotationWrapper:
    """
    Uploads a given filepath to omero as an annotation for parent_obj under namespace.

    parent_obj: omero.gateway.ParentWrapper
        Object that should own the annotation. (typically an ImageWrapper)
    file_path: str
        Local path of file to upload as annotation.
    namespace: str
        Remote namespace to put the file annotation
    mime: str, optional
        Mimetype for filetype. If None this will be assumed based on file extension
    overwrite: bool, Default: True
        Overwrites existing file annotation in this namespace.
    return: omero.gateway.FileAnnotationWrapper
        Uploaded FileAnnotation object.
    """
    conn = parent_obj._conn  # pylint: disable=W0212

    # if no mime provided try to parse from filename, if cannot, assume plaintext
    if mime is None:
        mime = lavlab.ctx.FILETYPE_ENUM.get_mimetype_from_path(file_path)

    # if overwrite is true and an annotation already exists in this namespace, delete it
    if overwrite is True:
        obj = parent_obj.getAnnotation(namespace)
        if obj is not None:
            conn.deleteObjects("Annotation", [obj.id], wait=True)

    # create, link, and return new annotation
    annot_obj = conn.createFileAnnfromLocalFile(file_path, mimetype=mime, ns=namespace)
    parent_obj.linkAnnotation(annot_obj)
    return annot_obj
