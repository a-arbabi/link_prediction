import xml.etree.cElementTree as ET
from urllib.request import urlopen
import gzip
import io

def process_xml_file(path):
  with gzip.open(path, 'rb') as xml_file:
    context = ET.iterparse(xml_file, events=("start", "end"))
    context = iter(context)
    event, root = next(context)
  
    for event, elem in context:
      if event == "end" and elem.tag == "Abstract":
        abstract_text = ' '.join(
            [x.strip() for x in elem.itertext()
            if len(x.strip())>0]).strip()
        yield abstract_text
        root.clear()

def process_xml_url(url):
  mysock = urlopen(url)
  memfile = io.BytesIO(mysock.read())
  for abstract in process_xml_file(memfile):
    yield abstract

def main():
  output_path = '/home/ubuntu/pubmed/pubmed_abstracts.txt'
  
  with open(output_path, 'w') as f:
    for shard_id in range(800, 1016):
      print(shard_id)
      url_base = 'ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline/'
      file_name = 'pubmed20n' + "{:04d}".format(shard_id) + '.xml.gz'
      url = url_base + file_name
  
      for i, abstract in enumerate(process_xml_url(url)):
        f.write(abstract)
        f.write('\n')
      f.flush()

if __name__ == '__main__':
  main()
