"""
Hardware control for Meadowlark SLMs.
Tested with Meadowlark
Updated to support the latest Blink SDK API in standard mode.
"""
import os
import ctypes
import warnings
import numpy as np
from slm import SLM

DEFAULT_SDK_PATH = "C:\\Program Files\\Meadowlark Optics\\Blink OverDrive Plus\\"


class Meadowlark(SLM):
    """
    Interfaces with Meadowlark SLMs using standard mode.

    Attributes
    ----------
    slm_lib : ctypes.CDLL
        Connection to the Meadowlark library.
    image_lib : ctypes.CDLL
        Connection to the ImageGen library.
    sdk_path : str
        Path of the Blink SDK folder.
    board_number : ctypes.c_uint
        The SLM board number, typically 1.
    """

    def __init__(
        self,
        verbose=True,
        sdk_path=DEFAULT_SDK_PATH,
        lut_path=None,
        wav_um=1,
        pitch_um=(8, 8),
        **kwargs,
    ):
        r"""
        Initializes an instance of a Meadowlark SLM in standard mode.

        Arguments
        ---------
        verbose : bool
            Whether to print extra information.
        sdk_path : str
            Path of the Blink SDK installation folder.
        lut_path : str OR None
            Passed to :meth:`load_lut`. Looks for the voltage 'look-up table' data
            which is necessary to run the SLM.
        wav_um : float
            Wavelength of operation in microns. Defaults to 1 um.
        pitch_um : (float, float)
            Pixel pitch in microns. Defaults to 8 micron square pixels.
        **kwargs
            See :meth:`.SLM.__init__` for permissible options.
        """
        # Validates the DPI awareness of this context
        if verbose:
            print("Validating DPI awareness...", end="")
        awareness = ctypes.c_int()
        error_get = ctypes.windll.shcore.GetProcessDpiAwareness(
            0, ctypes.byref(awareness)
        )
        error_set = ctypes.windll.shcore.SetProcessDpiAwareness(2)
        success = ctypes.windll.user32.SetProcessDPIAware()
        if not success:
            raise RuntimeError(
                "Meadowlark failed to validate DPI awareness. "
                "Errors: get={}, set={}, awareness={}".format(
                    error_get, error_set, awareness.value
                )
            )
        if verbose:
            print("success")

        # Open the SLM libraries
        if verbose:
            print("Loading Blink SDK libraries...", end="")
        self.sdk_path = sdk_path
        blink_wrapper_path = os.path.join(sdk_path, "SDK", "Blink_C_wrapper")
        image_gen_path = os.path.join(sdk_path, "SDK", "ImageGen")

        try:
            ctypes.cdll.LoadLibrary(blink_wrapper_path)
            self.slm_lib = ctypes.CDLL("Blink_C_wrapper")

            # Check if ImageGen exists and load it if available
            if os.path.exists(image_gen_path) or os.path.exists(
                image_gen_path + ".dll"
            ):
                ctypes.cdll.LoadLibrary(image_gen_path)
                self.image_lib = ctypes.CDLL("ImageGen")
                self.has_image_gen = True
            else:
                self.has_image_gen = False
                if verbose:
                    print(
                        "(ImageGen library not found, pattern generation will be unavailable)",
                        end="",
                    )
        except Exception as e:
            print("failure")
            raise ImportError(
                f"Meadowlark libraries did not import correctly. "
                f"Is '{blink_wrapper_path}' the correct path? Error: {e}"
            )
        if verbose:
            print("success")

        # Initialize SDK parameters
        self.board_number = ctypes.c_uint(1)
        bit_depth = ctypes.c_uint(12)
        num_boards_found = ctypes.c_uint(0)
        constructed_okay = ctypes.c_bool(False)
        is_nematic_type = ctypes.c_bool(True)
        RAM_write_enable = ctypes.c_bool(True)
        use_GPU = ctypes.c_bool(True)
        max_transients = ctypes.c_uint(20)

        # Standard timing parameters
        self.wait_for_trigger = ctypes.c_uint(0)
        self.flip_immediate = ctypes.c_uint(0)  # Only used on 1024 models
        self.output_pulse_image_flip = ctypes.c_uint(0)
        self.output_pulse_image_refresh = ctypes.c_uint(0)
        self.timeout_ms = ctypes.c_uint(5000)

        # Initialize the standard SDK
        if verbose:
            print("Initializing SDK...", end="")
        self.slm_lib.Create_SDK(
            bit_depth,
            ctypes.byref(num_boards_found),
            ctypes.byref(constructed_okay),
            is_nematic_type,
            RAM_write_enable,
            use_GPU,
            max_transients,
            ctypes.c_uint(0),  # Use 0 for standard mode instead of LUT filename
        )

        self.isopen = True

        if not constructed_okay.value:
            # Check if we have access to error message function
            if hasattr(self.slm_lib, "Get_last_error_message"):
                self.slm_lib.Get_last_error_message.restype = ctypes.c_char_p
                error_msg = self.slm_lib.Get_last_error_message()
                error_str = error_msg.decode("utf-8") if error_msg else "Unknown error"
            else:
                error_str = "SDK construction failed"

            print("failure")
            raise RuntimeError(
                f"Blink SDK was not constructed successfully. Error: {error_str}"
            )

        if verbose:
            print(f"success\nFound {num_boards_found.value} SLM controller(s)")

        # Get SLM dimensions
        width = self.slm_lib.Get_image_width(self.board_number)
        height = self.slm_lib.Get_image_height(self.board_number)
        depth = self.slm_lib.Get_image_depth(self.board_number)

        # In standard mode, we need to load a LUT file
        if verbose:
            print("Loading LUT file...", end="")
        try:
            true_lut_path = self.load_lut(lut_path)
            if verbose:
                if true_lut_path != lut_path:
                    print(f"success\n(loaded from '{true_lut_path}')")
                else:
                    print("success")
        except RuntimeError as e:
            if verbose:
                print("failure\n(could not find .lut file)")
            raise e

        # Construct other variables
        super().__init__(
            (width, height),
            bitdepth=depth,
            name=kwargs.pop("name", "Meadowlark"),
            wav_um=wav_um,
            pitch_um=pitch_um,
            **kwargs,
        )

        if self.bitdepth > 8:
            warnings.warn(
                f"Bitdepth of {self.bitdepth} > 8 detected; "
                "verify that your settings are correct."
            )

        # Initialize with blank pattern
        self.set_phase(None)

    def load_lut(self, lut_path=None):
        """
        Loads a voltage 'look-up table' (LUT) to the SLM.
        This converts requested phase values to physical voltage perturbing
        the liquid crystals.

        Parameters
        ----------
        lut_path : str OR None
            Path to look for an LUT file in.
            -   If this is a .lut file, then this file is loaded to the SLM.
            -   If this is a directory, then searches all files inside the
                directory, and loads either the alphabetically-first .lut file
                or if possible the alphabetically-first .lut file starting with ``"slm"``
                which is more likely to correspond to the LUT customized to an SLM.

        Raises
        ------
        RuntimeError
            If a .lut file is not found.

        Returns
        -------
        str
            The path which was used to load the LUT.
        """
        # If a path is not given, search inside the SDK path
        if lut_path is None:
            lut_path = os.path.join(self.sdk_path, "LUT Files")

        # If we already have a .lut file, proceed.
        if len(lut_path) > 4 and lut_path[-4:].lower() == ".lut":
            pass
        else:  # Otherwise, treat the path like a folder and search inside the folder.
            lut_file = None

            # Check if directory exists
            if not os.path.exists(lut_path) or not os.path.isdir(lut_path):
                raise RuntimeError(
                    f"LUT directory '{lut_path}' does not exist or is not a directory"
                )

            for file in os.listdir(lut_path):
                # Only examine .lut files.
                if len(file) >= 4 and file[-4:].lower() == ".lut":
                    # Choose the first one.
                    if lut_file is None:
                        lut_file = file
                    # Or choose the first one that starts with "slm"
                    if file[:3].lower() == "slm" and not lut_file[:3].lower() == "slm":
                        lut_file = file
                        break

            # Throw an error if we didn't find a .lut file.
            if lut_file is not None:
                lut_path = os.path.join(lut_path, lut_file)
            else:
                raise RuntimeError(f"Could not find a .lut file at path '{lut_path}'")

        # Finally, load the lookup table
        # Convert the path to bytes for ctypes if needed
        if isinstance(lut_path, str):
            lut_path_bytes = lut_path.encode("utf-8")
        else:
            lut_path_bytes = lut_path

        self.slm_lib.Load_LUT_file(self.board_number, lut_path_bytes)
        return lut_path

    @staticmethod
    def info(verbose=True):
        """
        The normal behavior of this function is to discover the names of all the displays
        to help the user identify the correct display. However, Meadowlark software does
        not currently support multiple SLMs, so this function instead raises an error.

        Parameters
        ----------
        verbose : bool
            Whether to print the discovered information.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError(
            "Meadowlark software does not currently support multiple SLMs, "
            "so a function to identify SLMs is moot. "
            "If functionality with multiple SLMs is desired, contact them directly."
        )

    def close(self):
        """
        Clean up and close the connection to the SLM.
        See :meth:`.SLM.close`.
        """
        self.isopen = False
        self.slm_lib.Delete_SDK()

    def _set_phase_hw(self, display):
        """
        Sends the phase pattern to the SLM hardware in standard mode.
        Implementation of abstract method from SLM base class.

        Parameters
        ----------
        display : numpy.ndarray
            The phase pattern to display on the SLM.
        """
        # Calculate the size of the image in bytes
        bytes_per_pixel = self.bitdepth // 8
        if bytes_per_pixel < 1:
            bytes_per_pixel = 1

        total_bytes = display.size * bytes_per_pixel

        # Write the image using standard mode
        ret_val = self.slm_lib.Write_image(
            self.board_number,
            display.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            total_bytes,
            self.wait_for_trigger,
            self.flip_immediate,
            self.output_pulse_image_flip,
            self.output_pulse_image_refresh,
            self.timeout_ms,
        )

        if ret_val == -1:
            raise RuntimeError("Failed to write image to SLM (DMA failed)")

        # Check if the SLM is ready for the next image
        ret_val = self.slm_lib.ImageWriteComplete(self.board_number, self.timeout_ms)
        if ret_val == -1:
            warnings.warn("SLM may not be ready for next image (trigger issue?)")

    ### Additional Meadowlark-specific functionality
    def get_temperature(self):
        """
        Read the temperature of the SLM.

        Returns
        -------
        float
            Temperature in degrees Celsius.
        """
        return self.slm_lib.Read_SLM_temperature(self.board_number)

    def generate_pattern(self, pattern_type, **kwargs):
        """
        Generate common phase patterns using the ImageGen library.

        Parameters
        ----------
        pattern_type : str
            Type of pattern to generate ('LG' for Laguerre-Gaussian, etc.)
        **kwargs
            Pattern-specific parameters

        Returns
        -------
        numpy.ndarray
            The generated pattern

        Raises
        ------
        RuntimeError
            If ImageGen library is not available or pattern type is not supported.
        """
        if not self.has_image_gen:
            raise RuntimeError(
                "ImageGen library not available. Pattern generation not possible."
            )

        width = self.slm_lib.Get_image_width(self.board_number)
        height = self.slm_lib.Get_image_height(self.board_number)
        depth = self.slm_lib.Get_image_depth(self.board_number)

        # Create array for the pattern
        pattern = np.zeros(width * height, dtype=np.uint8)

        if pattern_type.lower() == "lg":
            # Laguerre-Gaussian pattern
            charge = kwargs.get("charge", 1)
            center_x = kwargs.get("center_x", width // 2)
            center_y = kwargs.get("center_y", height // 2)
            fork = kwargs.get("fork", 0)
            rgb = kwargs.get("rgb", 0)

            # Create empty array for wavefront correction if needed by the API
            wfc = np.zeros(width * height, dtype=np.uint8)

            try:
                # Try newer API first
                self.image_lib.Generate_LG(
                    pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    width,
                    height,
                    ctypes.c_uint(charge),
                    ctypes.c_uint(center_x),
                    ctypes.c_uint(center_y),
                    ctypes.c_uint(fork),
                )
            except Exception:
                # Fall back to older API that requires wavefront correction
                self.image_lib.Generate_LG(
                    pattern.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    wfc.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                    width,
                    height,
                    depth,
                    ctypes.c_uint(charge),
                    ctypes.c_uint(center_x),
                    ctypes.c_uint(center_y),
                    ctypes.c_uint(fork),
                    ctypes.c_uint(rgb),
                )
        else:
            raise ValueError(f"Pattern type '{pattern_type}' not supported")

        # Reshape to 2D array
        return pattern.reshape((height, width))
