import argparse, time, os
import yaml


class Workflow :
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        workflow_config = self.read_from_yaml(cfg.workflow_yaml)
        self.cluster_config = workflow_config['nodes'] 


    def read_from_yaml(self, file: str):
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Not found: {file}")
        with open(file, "r", encoding="utf-8") as stream:
            return yaml.safe_load(stream)
        

    def get_head_node(self) :
        head_node = None
        for node in self.cluster_config:
            if node["type"] == "head" :
                head_node = node
        return head_node
    

    def startup_ray_cluster(self) :
        for node in self.cluster_config:
            if node["type"] == "head" :
                ret = os.system(f'./run-ray-cluster.sh -r startup_head -a {node["node"]} \
                        -c {node["cores"]} \
                        -w {node["workspace_dir"]} \
                        -m {node["model_dir"]} \
                        -t {node["tmp_dir"]} \
                        -i {node["image"]}')
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
                        -w {node["workspace_dir"]} \
                        -m {node["model_dir"]} \
                        -t {node["tmp_dir"]} \
                        -u {node["user"]} \
                        -p {node["password"]} \
                        -i {node["image"]} \
                        -s {node["node"]}')
                if ret == 0 :
                    print("Successfully startup the workers!")
                else:
                    print("Startup the workers failed!")
                    return False
        return True
                

    def exec_pipeline(self, pipeline: str) :
        head_node = self.get_head_node()
        if head_node == None:
            print("head_node cannot be None!!")
            return
        ret = os.system(f'./run-ray-cluster.sh -r exec_pipeline -l {pipeline} -t {self.cfg.enable_sample}')
        print(f'ret={ret}')       


    def run_pipelines(self) :
        
        for pipeline in self.pipelines_config :
            config_pipeline = pipeline["name"].split("/")[-1] 
            exec_pipeline = self.cfg.pipeline_yaml.split("/")[-1]
            if self.cfg.pipeline_yaml == "all" or config_pipeline == exec_pipeline :
                ret = self.stop_database()
                if ret == 0 :
                    print("Clean the database container successfully!")
                else:
                    print("There is no database to be stopped!")
                ret = self.startup_database(pipeline["database"])
                if ret == 0 :
                    print("Startup the database container successfully!")
                    self.exec_pipeline(config_pipeline)
                else:
                    print("Failed to startup the database containers")

    

def parse_cmd():
    args = argparse.ArgumentParser(description='parse arguments', epilog=' ', formatter_class=argparse.RawTextHelpFormatter)
    args.add_argument('-w', required=True, type=str, dest='workflow_yaml', help='workflow config file')
    return args.parse_args()


if __name__ == "__main__":
    config = parse_cmd()
    workflow= Workflow(config)

    workflow.startup_ray_cluster()
    
        