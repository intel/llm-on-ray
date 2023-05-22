import argparse, time, os
import yaml


class Workflow :
    def __init__(self, cfg):
        self.cfg = cfg
        workflow_config = self.read_from_yaml(cfg.workflow_yaml)
        self.run_ray_cluster = workflow_config['general']['run_ray_cluster']
        self.run_hdfs = workflow_config['general']["run_hdfs"]
        self.run_training_job = workflow_config['general']['run_training_job']
        self.model_dir = workflow_config['general']['model_dir']
        self.tmp_dir = workflow_config['general']['tmp_dir']
        self.workspace_dir = workflow_config['general']['workspace_dir']
        self.image_name = workflow_config['general']['image_name']
        self.cluster_config = workflow_config['nodes'] 
        self.training_spec = workflow_config['training_spec']
        self.head_ip = self.get_head_node()['node']
        self.num_nodes = self.get_worker_nodes()[0]+1
        self.worker_ips = self.get_worker_nodes()[1]


    def read_from_yaml(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Not found: {file}")
        with open(file, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
        

    def get_head_node(self):
        head_node = None
        for node in self.cluster_config:
            if node["type"] == "head" :
                head_node = node
        return head_node


    def get_worker_nodes(self):
        node_ips = []
        for node in self.cluster_config:
            if node['type'] == 'worker':
                node_ips.append(node['node'])
        return (len(node_ips), "@".join(node_ips))        


    def startup_ray_cluster(self) :
        for node in self.cluster_config:
            if node["type"] == "head" :
                ret = os.system(f'./run-ray-cluster.sh -r startup_head -a {node["node"]} \
                        -c {node["cores"]} \
                        -w {self.workspace_dir} \
                        -m {self.model_dir} \
                        -t {self.tmp_dir} \
                        -i {self.image_name}')
                if ret == 0 :
                    print("Successfully startup the ray head!")
                else:
                    print("Startup the ray head failed!")
                    return False
            else:
                head_node = self.get_head_node()
                if head_node == None:
                    print("head_node cannot be None!!")
                    return False
                
                ret = os.system(f'./run-ray-cluster.sh -r startup_worker -a {head_node["node"]} \
                        -c {node["cores"]} \
                        -w {self.workspace_dir} \
                        -m {self.model_dir} \
                        -t {self.tmp_dir} \
                        -u {node["user"]} \
                        -p {node["password"]} \
                        -i {self.image_name} \
                        -s {node["node"]}')
                if ret == 0 :
                    print("Successfully startup the workers!")
                else:
                    print("Startup the workers failed!")
                    return False
        return True
                
    def startup_hdfs(self):

        ret = os.system(f'docker exec ray-leader bash run-hdfs.sh -m {self.head_ip} -w {self.worker_ips}')
            
        if ret == 0:
            print("Successfully startup HDFS!")
        else:
            print("Startup HDFS failed!")
            return False

        return True
    
    def run_training(self):
        
        if self.training_spec['task_name'] == 'clm':
            model_name = self.training_spec['model_name']
            dataset_name = self.training_spec['dataset_name']
            dataset_config = self.training_spec['dataset_config']
            per_device_train_batch_size = self.training_spec['per_device_train_batch_size']
            per_device_eval_batch_size = self.training_spec['per_device_eval_batch_size']
            num_train_epochs = self.training_spec['num_train_epochs']

            ret = os.system(f'docker exec ray-leader python -u Finetune/run_clm_no_trainer_ray.py \
                            --model_name_or_path {model_name} \
                            --dataset_name {dataset_name} \
                            --dataset_config_name {dataset_config} \
                            --per_device_train_batch_size {per_device_train_batch_size} \
                            --per_device_eval_batch_size {per_device_eval_batch_size} \
                            --num_train_epochs {num_train_epochs} \
                            --address {self.head_ip} \
                            --num_workers {self.num_nodes}')

            if ret == 0:
                print("Training Job finished!")
            else:
                print("Training Job failed! Please check the log for debugging.")

        else:
            raise ValueError("only clm training taks is supported.")

    def process(self):
        if self.run_ray_cluster:
            self.startup_ray_cluster()
        
        if self.run_hdfs:
            self.startup_hdfs()

        if self.run_training_job:
            self.run_training()
        

def parse_cmd():
    args = argparse.ArgumentParser(description='parse arguments', epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-w', required=True, type=str, dest='workflow_yaml', help='workflow config file')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    workflow = Workflow(config)
    workflow.process()
    
        