from .rat import *
import cv2

# def crop_image(image):
#     boxes = process_img(image)
#     print(boxes)
#     # left_hand
#     x_min, y_min, x_max, y_max = boxes[0]
#     lh_cropped_image = image[x_min : x_max, y_min : y_max]
#     target_size = (256, 192)
#     lh_cropped_image = cv2.resize(lh_cropped_image, target_size)

#     # right_hand
#     x_min, y_min, x_max, y_max = boxes[1]
#     rh_cropped_image = image[x_min : x_max, y_min : y_max]
#     target_size = (256, 192)
#     rh_cropped_image = cv2.resize(rh_cropped_image, target_size)

#     return lh_cropped_image, rh_cropped_image

def crop_image(image):
    boxes, hand_sides = process_img(image)  # Unpack the tuple returned by process_img
    if boxes is None or hand_sides is None:
        raise ValueError("No hands detected in the image")

    print(boxes)

    lh_cropped_image = None
    rh_cropped_image = None
    target_size = (256, 192)

    # Iterate over the boxes and hand_sides to process both hands
    for i, (box, is_right) in enumerate(zip(boxes, hand_sides)):
        x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers for image slicing

        if is_right:
            rh_cropped_image = image[y_min:y_max, x_min:x_max]  # Right hand
            rh_cropped_image = cv2.resize(rh_cropped_image, target_size)
        else:
            lh_cropped_image = image[y_min:y_max, x_min:x_max]  # Left hand
            lh_cropped_image = cv2.resize(lh_cropped_image, target_size)

    return lh_cropped_image, rh_cropped_image


def save_cropped_images(image, output_dir):
    # Crop the images
    lh_image, rh_image = crop_image(image)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the left-hand image
    lh_image_path = os.path.join(output_dir, "left_hand.png")
    cv2.imwrite(lh_image_path, lh_image)
    print(f"Left hand image saved to {lh_image_path}")

    # Save the right-hand image
    rh_image_path = os.path.join(output_dir, "right_hand.png")
    cv2.imwrite(rh_image_path, rh_image)
    print(f"Right hand image saved to {rh_image_path}")

if __name__ == "__main__":
    img_path = "/home/sjtu/eccv_workspace/hold/code/data/arctic_s03_box_grab_01_1/build/image/0001.png"
    output_dir = "/home/sjtu/eccv_workspace/hamer/hamer/models/components/test_crop"  # Specify your output directory here

    # Read the image
    image = cv2.imread(img_path)

    # Save the cropped images to the specified directory
    save_cropped_images(image, output_dir)

# if __name__ == "__main__":
#     img_path = "/home/sjtu/eccv_workspace/hold/code/data/arctic_s03_box_grab_01_1/build/image/0000.png"
#     image = cv2.imread(img_path)
#     lh_image, rh_image = crop_image(image)

#     # Display the images using OpenCV to verify
#     cv2.imshow("Left Hand", lh_image)
#     cv2.imshow("Right Hand", rh_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()




