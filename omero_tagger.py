import yaml
import os
from omero.gateway import BlitzGateway, TagAnnotationWrapper

# Load configurations
def load_config(config_file):
    with open(config_file, 'r') as stream:
        return yaml.safe_load(stream)

# Connect to OMERO
def connect_to_omero(config):
    username = config["config"]["sa_username"]
    password = config["config"]["sa_password"]
    conn = BlitzGateway(username, password, host='lavlab.mcw.edu', secure=True)
    conn.connect()
    return conn

# Create and link a new tag annotation
def tag_image(conn, image, tag_value, tag_desc=""):
    tag_ann = TagAnnotationWrapper(conn)
    tag_ann.setDescription("Autotag: "+tag_desc)
    tag_ann.setValue(tag_value)
    print(tag_value)
    # tag_ann.save()
    # image.linkAnnotation(tag_ann)

def parse_and_tag_image_name(conn, image, config):
    name, _ = os.path.splitext(image.getName()) # strip extension
    if name ==~ "\.ome":
        name = name.removesuffix('.ome')
    name_parts = name.split('_')
    for part in name_parts:
        if part in config['images']['name']['map']:
            tag_image(conn, image, config['images']['name']['map'][part])
        else:
            tag_image(conn, image, part)

# Check the count of ROIs and tag accordingly
def check_roi_and_tag(conn, image, config):
    rois_count = len(list(image.getRois()))
    for condition, annotations in config['images']['annotations']['rois'].items():
        for tag, count in annotations.items():
            if (condition == 'gt' and rois_count > count) or \
               (condition == 'le' and rois_count <= count) or \
               (condition == 'eq' and rois_count == count):
                tag_image(conn, image, tag)

# Process all images based on the YAML configurations
def process_images(conn, config):
    images = list(conn.getObjects("Image"))
    for image in images:
        print(image.getName())
        parse_and_tag_image_name(conn, image, config)
        check_roi_and_tag(conn, image, config)


def main():
    config = load_config('tagger.yml')
    conn = connect_to_omero(config)
    process_images(conn, config)
    conn.close()

if __name__ == "__main__":
    main()
