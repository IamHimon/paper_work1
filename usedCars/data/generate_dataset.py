from lxml import etree
import xml.dom.minidom
import random
import pandas as pd

names = ['Brand', 'Price', 'Vehicle', 'Odometer', 'Colour', 'Transmission', 'Body', 'Engine', 'Fuel Enconomy']


def fast_iter(xml_write):
    test_df = pd.read_csv('test_data_split_brand.txt', names=names).dropna()
    test_df['Odometer'] = test_df['Odometer'].apply(lambda x: str(x))

    doc = xml.dom.minidom.Document()
    root = doc.createElement('Cars')
    root.setAttribute('Source', 'salsecar')
    doc.appendChild(root)

    # print(test_df)
    count = 0
    for index, row in test_df.iterrows():
        count += 1
        print('----')

        NodeCar = doc.createElement('Car')
        node_Brand = doc.createElement('Brand')
        node_Price = doc.createElement('Price')
        node_Vehicle = doc.createElement('Vehicle')
        node_Odometer = doc.createElement('Odometer')
        node_Colour = doc.createElement('Colour')
        node_Transmission = doc.createElement('Transmission')
        node_Body = doc.createElement('Body')
        node_Engine = doc.createElement('Engine')
        node_Fuel_Enconomy = doc.createElement('Fuel_Enconomy')
        node_record_ID = doc.createElement('ID')


        node_Brand.appendChild(doc.createTextNode(row['Brand']))
        node_Price.appendChild(doc.createTextNode(row['Price']))
        node_Vehicle.appendChild(doc.createTextNode(row['Vehicle']))
        node_Odometer.appendChild(doc.createTextNode(row['Odometer']))
        node_Colour.appendChild(doc.createTextNode(row['Colour']))
        node_Transmission.appendChild(doc.createTextNode(row['Transmission']))
        node_Body.appendChild(doc.createTextNode(row['Body']))
        node_Engine.appendChild(doc.createTextNode(row['Engine']))
        node_Fuel_Enconomy.appendChild(doc.createTextNode(row['Fuel Enconomy']))
        node_record_ID.appendChild(doc.createTextNode(str(count)))


        NodeCar.appendChild(node_Brand)
        NodeCar.appendChild(node_Price)
        NodeCar.appendChild(node_Vehicle)
        NodeCar.appendChild(node_Odometer)
        NodeCar.appendChild(node_Colour)
        NodeCar.appendChild(node_Transmission)
        NodeCar.appendChild(node_Body)
        NodeCar.appendChild(node_Engine)
        NodeCar.appendChild(node_Fuel_Enconomy)
        NodeCar.appendChild(node_record_ID)

        root.appendChild(NodeCar)

    doc.writexml(xml_write, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


if __name__ == '__main__':
    xml_write = open('uc_template_data.xml', 'w+')
    fast_iter(xml_write)



