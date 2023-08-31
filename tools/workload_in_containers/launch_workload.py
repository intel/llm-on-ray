import argparse, time, os
import yaml


class Workflow :
    def __init__(self, cfg):
        self.cfg = cfg
        workflow_config = self.read_from_yaml(cfg.workflow_yaml)
        self.run_ray_cluster = workflow_config['general']['run_ray_cluster']
        self.run_data_processing = workflow_config['general']['run_data_processing']
        self.hf_dir = workflow_config['general']['hf_dir']
        self.shared_dir = workflow_config['general']['shared_dir']
        self.local_dir = workflow_config['general']['local_dir']
        self.workspace_dir = workflow_config['general']['workspace_dir']
        self.image_name = workflow_config['general']['image_name']
        self.cluster_config = workflow_config['nodes'] 
        self.data_processing_spec = workflow_config['data_processing_spec']
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
                        -m {self.hf_dir} \
                        -t {self.shared_dir} \
                        -l {self.local_dir} \
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
                        -m {self.hf_dir} \
                        -t {self.shared_dir} \
                        -l {self.local_dir} \
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
                

    def run_dp(self):
        pass 


    def process(self):
        if self.run_ray_cluster:
            self.startup_ray_cluster()

        if self.run_data_processing:
            self.run_dp()
        
        
def parse_cmd():
    args = argparse.ArgumentParser(description='parse arguments', epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-w', required=True, type=str, dest='workflow_yaml', help='workflow config file')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    workflow = Workflow(config)
    workflow.process()
    
        
