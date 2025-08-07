#!/usr/bin/env python3
"""
Example script demonstrating XNAT support in lavlab-python-utils.

This script shows how to:
1. Configure XNAT service
2. Connect to XNAT server
3. List projects, subjects, and experiments
4. Download scan files

Requirements:
- lavlab-python-utils[xnat] package installed
- XNAT server configuration in ~/.lavlab.yml
"""

import lavlab
from lavlab.xnat import connect
from lavlab.xnat.helpers import (
    get_projects,
    get_subjects, 
    get_experiments,
    get_scan_files,
    download_scan_file
)


def main():
    """Demonstrate XNAT functionality."""
    
    # Configure for XNAT service - this would typically be in ~/.lavlab.yml
    # Example configuration:
    config = {
        "histology": {
            "service": {
                "name": "xnat",
                "host": "https://your-xnat-server.org",
                # username/password will be prompted or loaded from keyring
            }
        }
    }
    
    print("XNAT Support Example")
    print("=" * 50)
    
    try:
        # Connect to XNAT server
        print("Connecting to XNAT server...")
        session = connect()
        print("✓ Connected successfully!")
        
        # List available projects
        print("\nAvailable projects:")
        projects = get_projects(session)
        for i, project_id in enumerate(projects[:5]):  # Show first 5
            print(f"  {i+1}. {project_id}")
        
        if projects:
            # Use first project as example
            project_id = projects[0]
            print(f"\nExploring project: {project_id}")
            
            # List subjects in the project
            subjects = get_subjects(session, project_id)
            print(f"  Subjects found: {len(subjects)}")
            
            if subjects:
                # Use first subject as example
                subject_id = subjects[0] 
                print(f"  Exploring subject: {subject_id}")
                
                # List experiments for the subject
                experiments = get_experiments(session, project_id, subject_id)
                print(f"    Experiments found: {len(experiments)}")
                
                if experiments:
                    # Show experiment details
                    experiment_id = experiments[0]
                    print(f"    Example experiment: {experiment_id}")
                    
                    # This would typically be used for actual file downloads:
                    # with download_scan_file(session, experiment_id, scan_id, filename) as file_path:
                    #     print(f"Downloaded file: {file_path}")
                    #     # Process the file here
                    
        print("\n✓ XNAT exploration completed successfully!")
        
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nMake sure to configure XNAT service in ~/.lavlab.yml:")
        print("""
histology:
  service:
    name: 'xnat'
    host: 'https://your-xnat-server.org'
    # username and password will be prompted or loaded from keyring
""")
    
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
    
    finally:
        # Clean up connection
        try:
            session.disconnect()
            print("✓ Disconnected from XNAT server")
        except:
            pass


if __name__ == "__main__":
    main()