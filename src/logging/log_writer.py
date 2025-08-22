import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

class LogWriter:
    """
    Keeps track of everything that happens in a CSV file.
    Records when stuff was found, what state we're in, etc.
    Makes it way easier to figure out what went wrong later.
    Tries not to choke if the file gets locked or disk fills up.
    """
    
    # Define standard columns that will always be present
    STANDARD_COLUMNS = [
        'timestamp',
        'state',
        'bbox_x',
        'bbox_y',
        'bbox_width',
        'bbox_height',
        'angle',
    ]
    
    def __init__(self, csv_path: Union[str, Path]):
        """
        Initialize the log writer.
        
        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        self.current_columns = self.STANDARD_COLUMNS.copy()
        self._ensure_csv_exists()
        
    def _ensure_csv_exists(self):
        """Create CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.current_columns)
                
    def _update_columns(self, extra: Dict) -> List[str]:
        """
        Update columns list based on extra fields.
        Returns complete list of columns.
        """
        new_columns = set(extra.keys()) - set(self.current_columns)
        if new_columns:
            # Read existing data
            rows = []
            with open(self.csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            # Update columns list
            self.current_columns.extend(sorted(new_columns))
            
            # Rewrite file with new headers and data
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.current_columns)
                writer.writeheader()
                writer.writerows(rows)
                
        return self.current_columns
        
    def write_event(self, 
                    timestamp: Optional[datetime] = None,
                    state: str = '',
                    bbox: Optional[Tuple[float, float, float, float]] = None,
                    angle: Optional[float] = None,
                    extra: Optional[Dict] = None):
        """
        Write an event to the log file.
        
        Args:
            timestamp: Event timestamp (default: current time)
            state: Current system state
            bbox: Bounding box as (x, y, width, height)
            angle: Detected angle in degrees
            extra: Dictionary of additional fields to log
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Prepare row data
        row_data = {
            'timestamp': timestamp.isoformat(),
            'state': state,
            'bbox_x': bbox[0] if bbox else '',
            'bbox_y': bbox[1] if bbox else '',
            'bbox_width': bbox[2] if bbox else '',
            'bbox_height': bbox[3] if bbox else '',
            'angle': angle if angle is not None else '',
        }
        
        # Add extra fields
        if extra:
            # Convert any complex objects to JSON strings
            for key, value in extra.items():
                if isinstance(value, (dict, list)):
                    extra[key] = json.dumps(value)
            row_data.update(extra)
            
        # Ensure columns are up to date
        columns = self._update_columns(extra or {})
        
        # Write row safely
        try:
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(row_data)
        except Exception as e:
            print(f"Error writing to log: {str(e)}")
            
    def get_filepath(self) -> Path:
        """Get the current log file path."""
        return self.csv_path
