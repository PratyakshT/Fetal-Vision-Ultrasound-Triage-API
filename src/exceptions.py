class FetalVisionError(Exception):
    """Base exception for all Fetal-Vision pipeline errors."""
    pass

class MetadataMissingError(FetalVisionError):
    """Raised when an ultrasound image is missing required pixel-size metadata."""
    def __init__(self, filename, message="Missing pixel size metadata for normalization."):
        self.filename = filename
        self.message = f"{filename}: {message}"
        super().__init__(self.message)

class ResolutionMismatchError(FetalVisionError):
    """Raised when an image tensor falls outside the expected mathematical bounds."""
    def __init__(self, shape, expected_shape, message="Image resolution mismatch."):
        self.shape = shape
        self.expected_shape = expected_shape
        self.message = f"{message} Got {shape}, expected {expected_shape}."
        super().__init__(self.message)

class CorruptedImageError(FetalVisionError):
    """Raised when an uploaded file cannot be parsed or read properly."""
    def __init__(self, filename, message="File corrupted or unreadable."):
        self.filename = filename
        self.message = f"{filename}: {message}"
        super().__init__(self.message)