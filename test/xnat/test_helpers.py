# pylint: skip-file
# type: ignore
import pytest
from unittest.mock import Mock, patch
from lavlab.xnat.helpers import (
    get_projects,
    get_subjects,
    get_experiments,
    get_scan_files,
    download_scan_file,
    find_experiments_by_type,
    search_experiments
)


class TestXNATHelpers:
    """Test XNAT helper functions."""

    def test_get_projects(self, mock_xnat_session):
        """Test getting list of projects."""
        mock_xnat_session.projects = {"proj1": Mock(), "proj2": Mock()}
        
        projects = get_projects(mock_xnat_session)
        
        assert projects == ["proj1", "proj2"]

    def test_get_subjects(self, mock_xnat_session):
        """Test getting list of subjects for a project."""
        mock_project = Mock()
        mock_project.subjects = {"subj1": Mock(), "subj2": Mock()}
        mock_xnat_session.projects = {"proj1": mock_project}
        
        subjects = get_subjects(mock_xnat_session, "proj1")
        
        assert subjects == ["subj1", "subj2"]

    def test_get_experiments(self, mock_xnat_session):
        """Test getting list of experiments for a project and subject."""
        mock_experiment = Mock()
        mock_subject = Mock()
        mock_subject.experiments = {"exp1": mock_experiment, "exp2": mock_experiment}
        mock_project = Mock()
        mock_project.subjects = {"subj1": mock_subject}
        mock_xnat_session.projects = {"proj1": mock_project}
        
        experiments = get_experiments(mock_xnat_session, "proj1", "subj1")
        
        assert experiments == ["exp1", "exp2"]

    def test_get_scan_files(self, mock_xnat_session):
        """Test getting files for a scan."""
        mock_file1 = Mock()
        mock_file2 = Mock()
        mock_scan = Mock()
        mock_scan.files = {"file1.dcm": mock_file1, "file2.dcm": mock_file2}
        mock_experiment = Mock()
        mock_experiment.scans = {"scan1": mock_scan}
        mock_xnat_session.experiments = {"exp1": mock_experiment}
        
        files = get_scan_files(mock_xnat_session, "exp1", "scan1")
        
        assert files == [mock_file1, mock_file2]

    @patch('os.path.exists')
    @patch('os.remove')
    def test_download_scan_file(self, mock_remove, mock_exists, mock_xnat_session):
        """Test downloading and cleanup of scan file."""
        mock_file = Mock()
        mock_file.download.return_value = "/tmp/test_file.dcm"
        mock_scan = Mock()
        mock_scan.files = {"test.dcm": mock_file}
        mock_experiment = Mock()
        mock_experiment.scans = {"scan1": mock_scan}
        mock_xnat_session.experiments = {"exp1": mock_experiment}
        
        mock_exists.return_value = True
        
        with download_scan_file(mock_xnat_session, "exp1", "scan1", "test.dcm") as file_path:
            assert file_path == "/tmp/test_file.dcm"
            mock_file.download.assert_called_once()
        
        mock_remove.assert_called_once_with("/tmp/test_file.dcm")

    def test_find_experiments_by_type(self, mock_xnat_session):
        """Test finding experiments by type."""
        mock_exp1 = Mock()
        mock_exp1.attrs = {'xsi:type': 'xnat:mrSessionData'}
        mock_exp2 = Mock()
        mock_exp2.attrs = {'xsi:type': 'xnat:ctSessionData'}
        mock_exp3 = Mock()
        mock_exp3.attrs = {'xsi:type': 'xnat:mrSessionData'}
        
        mock_project = Mock()
        mock_project.experiments = {
            "exp1": mock_exp1,
            "exp2": mock_exp2, 
            "exp3": mock_exp3
        }
        mock_xnat_session.projects = {"proj1": mock_project}
        
        mr_experiments = find_experiments_by_type(mock_xnat_session, "proj1", "xnat:mrSessionData")
        
        assert len(mr_experiments) == 2
        assert mock_exp1 in mr_experiments
        assert mock_exp3 in mr_experiments
        assert mock_exp2 not in mr_experiments

    def test_search_experiments(self, mock_xnat_session):
        """Test searching experiments with criteria."""
        mock_table = Mock()
        mock_xnat_session.experiments.tabulate.return_value = mock_table
        
        result = search_experiments(mock_xnat_session, project="test_project")
        
        assert result == mock_table
        mock_xnat_session.experiments.tabulate.assert_called_once_with(project="test_project")