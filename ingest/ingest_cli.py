"""
Ingestion CLI for b_rag.

Usage:
    python -m ingest.ingest_cli [--dir data/raw] [--recreate]
    python -m ingest.ingest_cli --source google_drive --folder-id XXXXX [--recreate]

Supports multiple data sources through the connector system.
"""
import argparse
import sys
from pathlib import Path

from connectors import LocalFilesConnector, GoogleDriveConnector
from ingest.pipeline import IngestionPipeline, ingest


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into Qdrant for RAG"
    )
    
    # Source selection
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "google_drive"],
        help="Data source to ingest from (default: local)",
    )
    
    # Local file options
    parser.add_argument(
        "--dir",
        type=str,
        default="data/raw",
        help="Directory for local files (default: data/raw)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Single file to ingest (local source only)",
    )
    
    # Google Drive options
    parser.add_argument(
        "--credentials",
        type=str,
        help="Path to Google service account JSON file",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        help="Google Drive folder ID to sync",
    )
    
    # Common options
    parser.add_argument(
        "--tenant",
        type=str,
        default="default",
        help="Tenant ID for multi-tenancy (default: default)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate the Qdrant collection (deletes existing data)",
    )
    
    args = parser.parse_args()
    
    # Create appropriate connector
    if args.source == "local":
        if args.file:
            # Single file - create connector for parent directory
            path = Path(args.file)
            if not path.exists():
                print(f"Error: File not found: {path}")
                sys.exit(1)
            connector = LocalFilesConnector(
                directory=path.parent,
                tenant_id=args.tenant,
                extensions=(path.suffix,),
            )
            # Override to only process this file
            connector._single_file = path
        else:
            directory = Path(args.dir)
            if not directory.exists():
                print(f"Error: Directory not found: {directory}")
                sys.exit(1)
            connector = LocalFilesConnector(
                directory=directory,
                tenant_id=args.tenant,
            )
    
    elif args.source == "google_drive":
        if not args.credentials:
            print("Error: --credentials required for Google Drive source")
            sys.exit(1)
        connector = GoogleDriveConnector(
            credentials_path=args.credentials,
            folder_id=args.folder_id,
            tenant_id=args.tenant,
        )
    
    else:
        print(f"Unknown source: {args.source}")
        sys.exit(1)
    
    # Run ingestion
    stats = ingest(connector, recreate=args.recreate)
    
    # Exit with error if no documents processed
    if stats.documents_processed == 0:
        print("\nNo documents were ingested. Check the source path and file formats.")
        sys.exit(1)
    
    # Exit with warning if there were errors
    if stats.errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
