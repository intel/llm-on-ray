import json 
import time 
import argparse 
import os 

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def read_json(json_file):

    with open(json_file) as file:
        parsed_json = json.load(file)

    return parsed_json

def get_all_episodes(parsed_json):

    parsed_json = dict(sorted(parsed_json.items()))

    return parsed_json.keys()

def identify_common_episode(first_json, second_json):

    first_episodes = get_all_episodes(first_json)
    second_episodes = get_all_episodes(second_json)
    
    common_episodes = list(set(first_episodes).intersection(second_episodes))

    if len(common_episodes) == 0:
        print("the 2 trainings have no episode overlapped. Check your json file!")
        return -1 
    elif len(common_episodes) > 1:
        print("the 2 trainings have more than 1 overlapped episodes. Check your json files!")
        return -1
    else:
        return common_episodes[0]

def compare_training_states(json1, json2, step):
    
    step = f'step_{step}'

    data_result = json1[step]['data'] == json2[step]['data']
    lr_result = json1[step]['learning_rate'] == json2[step]['learning_rate']
    loss_result = json1[step]['loss'] == json2[step]['loss']

    return data_result, lr_result, loss_result 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default='/home/user/tmp/state',
        help="absolute path of the json files"
    )
    args = parser.parse_args()

    # read the json files
    state1 = read_json(os.path.join(args.file_path, 'stepwise_training_state.json'))
    state2 = read_json(os.path.join(args.file_path, 'stepwise_training_state_recovery.json'))

    # identify the overlapped episode 
    common_episode = identify_common_episode(state1, state2)
    print(f"the common episode of 2 trainings:  {common_episode}\n")
    
    # compare the different training states
    data_result, lr_result, loss_result = compare_training_states(state1[common_episode], state2[common_episode], 0)

    # print out the detailed comparison results
    print(f"Are the Data the same?\n{data_result}")
    print(f"Are the Learning Rate the same?\n{lr_result}")
    print(f"Are the Training Loss the same?\n{loss_result}")

    if data_result and lr_result and loss_result:
        print(f"{bcolors.OKGREEN}\nrecovery tests all passed!{bcolors.ENDC}")
    else:
        print(f"{bcolors.FAIL}recovery test failed! check the detailed log above.{bcolors.ENDC}")


if __name__ == "__main__":
    main()
