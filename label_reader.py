from pipeline import run


def extract_label_data(image_path):
    return run(image_path)


if __name__ == "__main__":
    image_path = "images/USPS_Shipping_Label.JPG"
    label_data = extract_label_data(image_path)
    print(label_data)
