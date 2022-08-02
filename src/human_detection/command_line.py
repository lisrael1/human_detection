from human_detection import sec_cam
from optparse import OptionParser


def args():
    help_text = """
            security camera 
        """
    parser = OptionParser(usage=help_text, version="%prog 1.01 beta")  # you will see version when adding --version
    parser.add_option("-c", "--config_file", dest="yml", type="str", help="yml configuration file", default="")
    options, _ = parser.parse_args()
    return options


def main():
    options = args()
    sec_cam.Flow(path_to_conf_yml_file=options.yml)

    print('done!')
