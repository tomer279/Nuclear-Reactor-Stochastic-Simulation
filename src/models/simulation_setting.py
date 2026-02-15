"""Written by Tomer279 with the assistance of Cursor.ai.

Simulation control and output management classes.

This module provides classes for managing simulation execution,
progress tracking, and file output configuration. These classes
handle the operational aspects of running nuclear reactor simulations.

Classes:
    OutputSettings: File naming and directory management
    ProgressSettings: Progress monitoring and display configuration
    SimulationControl: Unified simulation control interface

Key Features:
    - Flexible file naming conventions for simulation results
    - Configurable progress tracking intervals
    - Centralized simulation control parameters
    - Easy serialization and configuration management

Usage Examples:
    # Configure file output
    output = OutputSettings(
        output_prefix='fission_run',
        output_directory='./results'
    )

    # Set up progress tracking
    progress = ProgressSettings(
        show_progress=True,
        progress_interval=5  # Show every 5%
    )

    # Unified control interface
    control = SimulationControl(
        output_settings=output,
        progress_settings=progress,
        save_results=True
    )

    # Generate result paths
    filename = control.generate_result_filename(
        method='EM',
        dead_time_type='uniform',
        fission_value=33.95
    )

Note:
    This module complements core_parameters.py by handling the
    operational aspects of simulations rather than the physical
    parameters themselves.
"""

from typing import Dict, Optional


class OutputSettings:
    """
    Output and file management settings.

    This class groups together parameters related to file output and naming.
    It provides methods for generating filenames and managing output paths.

    Attributes
    ----------
    output_prefix : str
        Prefix for output filenames
    output_directory : str
        Directory for output files

    Public Methods
    --------------
    generate_filename(method, dead_time_type, fission_value)
        Generate filename
    get_full_path(filename)
        Get full file path
    to_dict()
        Convert to dictionary for serialization
    """

    def __init__(self,
                 output_prefix: str = 'f',
                 output_directory: str = './'):
        """
        Initialize output settings.

        Parameters
        ----------
        output_prefix : str, optional
            Prefix for output filenames. The default is 'f'.
        output_directory : str, optional
            Directory for output files. The default is './'.
        """
        self.output_prefix = output_prefix
        self.output_directory = output_directory

    def generate_filename(self, method: str, dead_time_type: str,
                          fission_value: float) -> str:
        """
        Generate filename for simulation results.

        Parameters
        ----------
        method : str
            Simulation method name
        dead_time_type : str
            Dead time distribution type
        fission_value : float
            Fission rate value

        Returns
        -------
        str
            Generated filename in format:
            {prefix}{fission}_{method}_{dead_time}.npy
        """
        return (f"{self.output_prefix}{fission_value}"
                f"_{method}_{dead_time_type}.npy")

    def get_full_path(self, filename: str) -> str:
        """
        Get full file path.

        Parameters
        ----------
        filename : str
            Filename to get full path for

        Returns
        -------
        str
            Full file path: {output_directory}/{filename}
        """
        return f"{self.output_directory}/{filename}"

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of output settings
        """
        return {
            'output_prefix': self.output_prefix,
            'output_directory': self.output_directory,
        }


class ProgressSettings:
    """
    Progress tracking and display settings.

    Groups together parameters related to progress monitoring.
    """

    def __init__(self,
                 show_progress: bool = True,
                 progress_interval: int = 10):
        """
        Initialize progress settings.

        Parameters
        ----------
        show_progress : bool, optional
            Whether to show progress updates
        progress_interval : int, optional
            Progress update interval (percentage)
        """
        self.show_progress = show_progress
        self.progress_interval = progress_interval

    def should_show_progress(self,
                             current_step: int,
                             total_steps: int) -> bool:
        """ Determine if progress should be shown at current step."""
        if not self.show_progress:
            return False
        if total_steps <= 0:
            return False

        progress_percent = (current_step / total_steps) * 100
        return progress_percent % self.progress_interval < (100 / total_steps)

    def get_progress_message(self,
                             current_step: int,
                             total_steps: int) -> str:
        """Get formatted progress message."""
        progress_percent = (current_step / total_steps) * 100
        return (f"Progress: {current_step}/{total_steps}"
                f" ({progress_percent:.1f}%)")

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'show_progress': self.show_progress,
            'progress_interval': self.progress_interval
        }


class SimulationControl:
    """
    Simulation control and output parameters.

    This class groups together output settings and progress settings to provide
    a unified interface for simulation control.
    It manages file output, progress tracking, and result saving.

    Attributes
    ----------
    output : OutputSettings
        Output and file management settings
    progress : ProgressSettings
        Progress tracking settings
    save_results : bool
        Whether to save simulation results

    Public Methods
    --------------
    configure_output(prefix, directory)
        Configure output settings
    configure_progress(show, interval)
        Configure progress settings
    generate_result_filename(method, dead_time_type, fission_value)
        Generate filename
    get_progress_message(current_step, total_steps)
        Get formatted progress message
    get_result_path(method, dead_time_type, fission_value)
        Get full result path
    should_show_progress(current_step, total_steps)
        Determine if progress should be shown
    to_dict()
        Convert to dictionary for serialization
    """

    def __init__(self,
                 output_settings: Optional[OutputSettings] = None,
                 progress_settings: Optional[ProgressSettings] = None,
                 save_results: bool = True):
        """
        Initialize simulation control parameters.

        Parameters
        ----------
        output_settings : OutputSettings, optional
            Output and file management settings. Uses defaults if None.
        progress_settings : ProgressSettings, optional
            Progress tracking settings. Uses defaults if None.
        save_results : bool, optional
            Whether to save simulation results. Default is True.
        """
        self.output = output_settings or OutputSettings()
        self.progress = progress_settings or ProgressSettings()
        self.save_results = save_results

    def configure_output(self, prefix: str, directory: str):
        """
        Configure output settings.

        Parameters
        ----------
        prefix : str
            Prefix for output filenames
        directory : str
            Directory for output files
        """
        self.output = OutputSettings(prefix, directory)

    def configure_progress(self, show: bool, interval: int):
        """
        Configure progress settings.

        Parameters
        ----------
        show : bool
            Whether to show progress updates
        interval : int
            Progress update interval (percentage)
        """
        self.progress = ProgressSettings(show, interval)

    def generate_result_filename(self,
                                 method: str,
                                 dead_time_type: str,
                                 fission_value: float) -> str:
        """
        Generate filename for simulation results.

        Parameters
        ----------
        method : str
            Simulation method name
        dead_time_type : str
            Dead time distribution type
        fission_value : float
            Fission rate value

        Returns
        -------
        str
            Generated filename with full path
        """
        filename = self.output.generate_filename(
            method,
            dead_time_type,
            fission_value)
        return self.output.get_full_path(filename)

    def get_progress_message(self,
                             current_step: float,
                             total_steps: float):
        """
        Get formatted progress message.

        Parameters
        ----------
        current_step : float
            Current simulation step
        total_steps : float
            Total number of simulation steps

        Returns
        -------
        str
            Formatted progress message
        """
        return self.progress.get_progress_message(current_step, total_steps)

    def get_result_path(self,
                        method: str,
                        dead_time_type: str,
                        fission_value: float):
        """
        Get full path for simulation results.

        Parameters
        ----------
        method : str
            Simulation method name
        dead_time_type : str
            Dead time distribution type
        fission_value : float
            Fission rate value

        Returns
        -------
        str
            Full path for simulation results
        """
        filename = self.generate_result_filename(method,
                                                 dead_time_type,
                                                 fission_value)
        return self.output.get_full_path(filename)

    def should_show_progress(self,
                             current_step: int,
                             total_steps: int) -> bool:
        """
        Determine if progress should be shown.

        Parameters
        ----------
        current_step : int
            Current simulation step
        total_steps : int
            Total number of simulation steps

        Returns
        -------
        bool
            True if progress should be shown, False otherwise
        """
        return self.progress.should_show_progress(current_step, total_steps)

    def to_dict(self) -> Dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        Dict
            Dictionary representation of simulation control parameters
        """
        return {
            'output_settings': self.output.to_dict(),
            'progress_settings': self.progress.to_dict(),
            'save_results': self.save_results
        }
