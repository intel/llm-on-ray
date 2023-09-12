import argparse, time, os
import yaml


class Workflow :
    def __init__(self, cfg):
        workflow_config = self.read_from_yaml(cfg.workflow_yaml)
        self.remove_containers = cfg.remove_containers
        self.cluster_config = workflow_config['nodes'] 
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


    def remove_ray_cluster(self) :
        for node in self.cluster_config:
            if node["type"] == "head" :
                ret = os.system(f'./run-ray-cluster.sh -r clean_ray_head -a {node["node"]}')
                if ret == 0 :
                    print(f"Successfully removed the ray-leader on node {node['node']}!")
                else:
                    print(f"Remove the ray-leader failed on node {node['node']}!")
                    return False
            else:
                head_node = self.get_head_node()
                if head_node == None:
                    print("head_node cannot be None!!")
                    return False
                
                ret = os.system(f'./run-ray-cluster.sh -r clean_ray_worker -a {head_node["node"]} \
                        -u {node["user"]} \
                        -p {node["password"]} \
                        -s {node["node"]}')
                if ret == 0 :
                    print(f"Successfully removed the ray-worker on node {node['node']}!")
                else:
                    print(f"Remove the ray-worker failed on node {node['node']}!")
                    return False
        return True

    def stop_ray_cluster(self) :
        for node in self.cluster_config:
            if node["type"] == "head" :
                ret = os.system(f'./run-ray-cluster.sh -r stop_ray_head -a {node["node"]}')
                if ret == 0 :
                    print(f"Successfully stopped the ray-leader on node {node['node']}!")
                else:
                    print(f"Stop the ray-leader failed on node {node['node']}!")
                    return False
            else:
                head_node = self.get_head_node()
                if head_node == None:
                    print("head_node cannot be None!!")
                    return False
                
                ret = os.system(f'./run-ray-cluster.sh -r stop_ray_worker -a {head_node["node"]} \
                        -u {node["user"]} \
                        -p {node["password"]} \
                        -s {node["node"]}')
                if ret == 0 :
                    print(f"Successfully stopped the ray-worker on node {node['node']}!")
                else:
                    print(f"Stop the ray-worker failed on node {node['node']}!")
                    return False
        return True
    
    def process(self):
        if self.remove_containers:
            self.remove_ray_cluster()
        else:
            self.stop_ray_cluster()

        
def parse_cmd():
    args = argparse.ArgumentParser(description='parse arguments', epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-w', required=True, type=str, dest='workflow_yaml', help='workflow config file')
    args.add_argument('-r', default=False, action='store_true', dest='remove_containers', help='whether to remove the containers')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    workflow = Workflow(config)
    workflow.process()
    
        