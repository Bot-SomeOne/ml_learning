# python create_pt_to_h5.py catdog_resnet18_best.pt --output catdog_resnet18_best.h5

import torch
import h5py
import argparse
import os
from pathlib import Path


def convert_pt_to_h5(pt_file_path, h5_file_path=None):
    """
    Convert PyTorch .pt file to HDF5 .h5 file
    
    Args:
        pt_file_path: Path to the .pt file
        h5_file_path: Path to save the .h5 file (optional)
    """
    # Load PyTorch file
    print(f"Loading PyTorch file: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')
    
    # Generate output path if not provided
    if h5_file_path is None:
        h5_file_path = str(Path(pt_file_path).with_suffix('.h5'))
    
    # Create HDF5 file
    print(f"Saving to HDF5 file: {h5_file_path}")
    with h5py.File(h5_file_path, 'w') as h5_file:
        if isinstance(data, dict):
            # If data is a dictionary, save each key-value pair
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    h5_file.create_dataset(key, data=value.numpy())
                else:
                    # Handle non-tensor data
                    h5_file.attrs[key] = str(value)
        elif isinstance(data, torch.Tensor):
            # If data is a single tensor
            h5_file.create_dataset('data', data=data.numpy())
        else:
            # Handle other types
            print(f"Warning: Unsupported data type {type(data)}")
            h5_file.attrs['data'] = str(data)
    
    print("Conversion completed successfully!")
    return h5_file_path


def main():
    parser = argparse.ArgumentParser(description='Convert PyTorch .pt files to HDF5 .h5 files')
    parser.add_argument('input', type=str, help='Input .pt file path')
    parser.add_argument('--output', '-o', type=str, default=None, 
                        help='Output .h5 file path (optional)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return
    
    # Convert file
    convert_pt_to_h5(args.input, args.output)


if __name__ == '__main__':
    main()