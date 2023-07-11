"""
this script is for grouping data files into multiple data folders 
with the episode names for pre-training on Ray.
"""

import os 
import argparse 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src-data-path', 
        required=True,
        default='/home/user/tmp/processed_json',
        help="the path where the data to be grouped is located"
    )
    parser.add_argument(
        '--des-data-path', 
        required=True,
        default='/home/user/tmp/train_data',
        help="the path where the data to be grouped should be stored"
    )
    parser.add_argument(
        '--test', 
        default=False, 
        action='store_true', 
        help="if specified, test pretrain data folder will be created for recovery validation"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="the number of folders to be created"
    )
    args = parser.parse_args() 
    src_data_path = args.src_data_path 
    des_data_path = args.des_data_path 
    num_steps = args.num_steps 

    json_files = [os.path.join(src_data_path, name) for name in os.listdir(src_data_path) if name.endswith('json')]
    json_files = sorted(json_files)
    num_files = len(json_files)
    print(num_files)
    
    if args.test:
        idx = 0
        for step in range(4):

            new_data_path = os.path.join(des_data_path, f"episode_{step}")

            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path, exist_ok=True)
            
            file_chunk = json_files[idx]
            file_name = os.path.basename(file_chunk)
            os.rename(file_chunk, os.path.join(new_data_path, file_name))

            idx += 1 
            print(f"folder {step} finalized!")

    else:
        num_files_per_folder = num_files // num_steps
        rest = num_files % num_steps
        print(num_files_per_folder)

        true_steps = num_steps+1 if rest > 0 else num_steps 

        for step in range(true_steps):
            new_data_path = os.path.join(des_data_path, f"episode_{step}")
            
            if not os.path.exists(new_data_path):
                os.makedirs(new_data_path, exist_ok=True)

            file_chunk = json_files[num_files_per_folder*step:num_files_per_folder*(step+1)]

            for file in file_chunk:
                file_name = os.path.basename(file)
                os.rename(file, os.path.join(new_data_path, file_name))
            
            print(f"folder {step} finalized!")


if __name__ == "__main__":
    main()
