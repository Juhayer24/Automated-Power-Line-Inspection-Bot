from typing import Literal, Optional

class DebounceState:
    """
    Handles switching between safe/hazard states but isn't too jumpy about it.
    Waits for a few frames of the same thing before changing its mind.
    Helps avoid false alarms from random blips in the detector.
    """
    
    States = Literal['SAFE', 'HAZARD']
    
    def __init__(self, initial: States = 'SAFE', 
                 safe_to_hazard_frames: int = 3,
                 hazard_to_safe_frames: int = 5):
        """
        Initialize the state machine.
        
        Args:
            initial: Initial state ('SAFE' or 'HAZARD')
            safe_to_hazard_frames: Number of consecutive hazard frames needed to switch from SAFE to HAZARD
            hazard_to_safe_frames: Number of consecutive safe frames needed to switch from HAZARD to SAFE
        """
        if initial not in ['SAFE', 'HAZARD']:
            raise ValueError("Initial state must be 'SAFE' or 'HAZARD'")
            
        self._current_state: DebounceState.States = initial
        self._safe_to_hazard_frames = max(1, safe_to_hazard_frames)
        self._hazard_to_safe_frames = max(1, hazard_to_safe_frames)
        
        # Counters for consecutive frames
        self._hazard_counter = 0
        self._safe_counter = 0
        
    @property
    def current_state(self) -> States:
        """Get the current state."""
        return self._current_state
        
    @property
    def consecutive_hazard_frames(self) -> int:
        """Get the number of consecutive hazard frames."""
        return self._hazard_counter
        
    @property
    def consecutive_safe_frames(self) -> int:
        """Get the number of consecutive safe frames."""
        return self._safe_counter
        
    def update(self, is_hazard_frame: bool) -> States:
        """
        Update state machine based on new frame classification.
        
        Args:
            is_hazard_frame: True if current frame shows hazard, False otherwise
            
        Returns:
            Current state after update
        """
        if is_hazard_frame:
            self._hazard_counter += 1
            self._safe_counter = 0
            
            # Check if we should transition to HAZARD
            if (self._current_state == 'SAFE' and 
                self._hazard_counter >= self._safe_to_hazard_frames):
                self._current_state = 'HAZARD'
                
        else:
            self._safe_counter += 1
            self._hazard_counter = 0
            
            # Check if we should transition to SAFE
            if (self._current_state == 'HAZARD' and 
                self._safe_counter >= self._hazard_to_safe_frames):
                self._current_state = 'SAFE'
                
        return self._current_state
        
    def reset(self, new_state: Optional[States] = None):
        """
        Reset the state machine.
        
        Args:
            new_state: Optional new state to set. If None, keeps current state.
        """
        if new_state is not None:
            if new_state not in ['SAFE', 'HAZARD']:
                raise ValueError("New state must be 'SAFE' or 'HAZARD'")
            self._current_state = new_state
            
        self._hazard_counter = 0
        self._safe_counter = 0
