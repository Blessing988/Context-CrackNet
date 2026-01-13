#!/usr/bin/env python
"""
Check installed versions of all dependencies used in Context-CrackNet.
Run this script to get the exact versions for requirements.txt.
"""

def get_version(module_name, import_name=None, version_attr='__version__'):
    """Try to import a module and get its version."""
    if import_name is None:
        import_name = module_name
    try:
        module = __import__(import_name)
        version = getattr(module, version_attr, 'unknown')
        return f"{module_name}=={version}"
    except ImportError:
        return f"{module_name}==NOT_INSTALLED"

if __name__ == "__main__":
    print("=" * 50)
    print("Context-CrackNet Dependency Versions")
    print("=" * 50)
    
    # Core dependencies
    print("\n# Core dependencies")
    print(get_version("torch"))
    print(get_version("torchvision"))
    
    # Segmentation models
    print("\n# Segmentation models")
    print(get_version("segmentation-models-pytorch", "segmentation_models_pytorch"))
    
    # Image processing
    print("\n# Image processing")
    print(get_version("opencv-python", "cv2"))
    print(get_version("Pillow", "PIL"))
    print(get_version("albumentations"))
    
    # Scientific computing
    print("\n# Scientific computing")
    print(get_version("numpy"))
    print(get_version("scipy"))
    
    # Configuration
    print("\n# Configuration")
    print(get_version("PyYAML", "yaml"))
    
    # Progress bars
    print("\n# Progress bars")  
    print(get_version("tqdm"))
    
    print("\n" + "=" * 50)
    print("Copy the versions above to update requirements.txt")
    print("=" * 50)
