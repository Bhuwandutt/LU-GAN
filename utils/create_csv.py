import os
import pandas
from tqdm import tqdm
import json
from lxml import etree as et


def find_parentImage(xmlfile):
    parser = et.XMLParser(recover=True)
    tree = et.parse(xmlfile, parser=parser)
    root = tree.getroot()
    image_names = []
    for image in root.findall("./parentImage"):
        image_names.append(image.attrib['id'])
    return image_names


def test_XML(xmlfile):
    parser = et.XMLParser(recover=True)
    tree = et.parse(xmlfile,parser = parser)
    raw_report = tree.find("./MedlineCitation/Article/Abstract")
    for child in raw_report:
        if child.text is None:
            return False
    return True


basepath = "/Users/bhuwandutt/PycharmProjects/BSProject"
imagedir = "/Users/bhuwandutt/PycharmProjects/BSProject/data/imgs"
report_root = '/Users/bhuwandutt/PycharmProjects/BSProject/data/reports/ecgen-radiology'

csv_dir = basepath + '/' + 'csv'
# to_reportdir = to_basepath + '/' + 'reports/ecgen-radiology'
#
# label_file = 'label.json'
# with open(label_file) as json_file:
#     label = json.load(json_file)

# id for patients
subject_ids = []
report_id = []
# The path of the image
image_paths = []
# id for the report
report_paths = []
# id for the image
dicom_ids = []

# direction = []

# For each chest report
for name in os.listdir(report_root) :
    if not name.startswith('.') and os.path.isfile(os.path.join(report_root, name)):

        id_ = os.path.splitext(name)[0]
        report_file = os.path.join(report_root, name)
        # If the report is with correct format
        if test_XML(report_file):
            image_ids = find_parentImage(report_file)
            img_names = [image_id + '.png' for image_id in image_ids]
            if len(img_names) == 2:
                report_paths.append(report_root + '/' + name)
                report_id.append(id_)
                for img_n in img_names:
                    image_id = img_n.split('.')[0]
                    img_path = imagedir + '/' + img_n
                    assert os.path.exists(img_path), "{} image not exists".format(img_path)
                    subject_ids.append(id_)
                    dicom_ids.append(image_id)
                    # direction.append(label[img_n])
                    image_paths.append(img_path + '/' + img_n)

image_data = {'subject_id': subject_ids,
              # 'study_id':subject_ids,
              'dicom_id': dicom_ids,
              'path': image_paths
              # 'direction': direction
            }

img_df = pandas.DataFrame(image_data)
img_df.to_csv(os.path.join(csv_dir, 'openi_images.csv'))

report_data = {'subject_id': report_id,
               'path': report_paths
               # 'study_id':subject_ids
               }

rp_df = pandas.DataFrame(report_data)

rp_df.to_csv(os.path.join(csv_dir, 'openi_reports.csv'))



