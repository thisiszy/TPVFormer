from pathlib import Path
from typing import List, Optional, Dict, Union

from pydantic import BaseModel, Field
from PIL import Image
import numpy as np


class PoseEstimation(BaseModel):
    """
    Represents the pose estimation data for an object.

    Attributes:
        obj_name (str): Name of the object model.
        w2obj_t (List[float]): Translation vector of model/object local coordinates in world coordinates.
        w2obj_r (List[float]): Rotation vector of model/object local coordinates in world coordinates.
        obj2center (Optional[List[float]]): Translation vector of object center in object coordinates.
        size (List[float]): Size dimensions [width, height, depth] ([x, y, z] of the object).
    """
    obj_name: str
    w2obj_t: List[float] = Field(..., min_items=3, max_items=3)
    w2obj_r: List[float] = Field(..., min_items=3, max_items=3)
    obj2center: Optional[List[float]] = Field(None, min_items=3, max_items=3)
    size: List[float] = Field(..., min_items=3, max_items=3)


class Segmentation(BaseModel):
    """
    Represents segmentation data for a single object.

    Attributes:
        category_id (str): Category ID of the object (e.g., 'box', 'person').
        object_id (int): Unique ID of the object.
        shortuuid (Optional[str]): Short UUID of the object.
        bbox (Optional[List[int]]): Bounding box coordinates [x, y, width, height].
        mask (Optional[List[int]]): RLE encoded mask, within the bounding box.
        pick_class (Optional[Dict[str, str]]): Pick class annotations with user IDs as keys.
        pred_pick_class (Optional[Dict[str, Union[str, float]]]): Predicted pick class.
        pose_estimation (Optional[PoseEstimation]): Object pose estimations.
    """
    category_id: str
    object_id: int
    shortuuid: Optional[str] = None
    bbox: Optional[List[int]] = Field(None, min_items=4, max_items=4)
    mask: Optional[List[int]] = None
    pick_class: Optional[Dict[str, str]] = None
    pred_pick_class: Optional[Dict[str, Union[str, float]]] = None
    pose_estimation: Optional[PoseEstimation] = None


class LabelData(BaseModel):
    """
    Represents the label data for a single image frame.

    Attributes:
        segmentations (List[Segmentation]): A list of segmentation data for each object in the frame.
    """
    segmentations: List[Segmentation]


class MetadataDetails(BaseModel):
    """
    Represents metadata for a single image frame.

    Attributes:
        camera_matrix (List[List[float]]): 3x3 camera intrinsic matrix.
        position (List[float]): 3D position coordinates [x, y, z] of the camera.
        rotation (List[float]): 3D rotation angles [x, y, z] of the camera.
        tags (Optional[List[str]]): Optional tags/labels associated with the view.
    """
    camera_matrix: List[List[float]] = Field(..., min_items=3, max_items=3)
    position: List[float] = Field(..., min_items=3, max_items=3)
    rotation: List[float] = Field(..., min_items=3, max_items=3)
    tags: Optional[List[str]] = None


class Metadata(BaseModel):
    """
    Represents metadata for a single image frame.
    """
    metadata: MetadataDetails


class FlinkDatapoint(BaseModel):
    """
    Represents a single datapoint in the Flink dataset.

    Attributes:
        image (Path): The image file path.
        depth (Optional[Path]): The depth map file path, if available.
        label_data (LabelData): The label data loaded from the corresponding JSON file.
        metadata (Metadata): The metadata loaded from the corresponding JSON file.
    """
    image: Path
    depth: Optional[Path] = None
    label_data: LabelData
    metadata: Metadata

    def get_image(self, type: str = "numpy") -> Union[Image.Image, np.ndarray]:
        """Get image in specified type."""
        return self._get_image_helper(self.image, type)

    def get_depth(self, type: str = "numpy") -> Union[Image.Image, np.ndarray]:
        """Get depth image in specified type."""
        return self._get_image_helper(self.depth, type)

    def _get_image_helper(self, path: Path, type: str) -> Union[Image.Image, np.ndarray]:
        """Helper function to load and convert images."""
        img = Image.open(path)

        if type == "PIL":
            return img
        if type == "numpy":
            return np.array(img)
        else:
            raise ValueError(f"Invalid type: {type}")


class FlinkDatasetLoader:
    """
    Loader for a single-view Flink dataset.

    Args:
        base_path (Union[str, Path]): The root directory of the single-view Flink dataset.

    Attributes:
        base_path (Path): The root directory of the dataset.
    """

    def __init__(self, base_path: str | Path):
        self.base_path = Path(base_path)
        if not self.base_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.base_path}")

        self.datapoints = self._load_datapoints()

    def __len__(self) -> int:
        """
        Returns the number of datapoints in the dataset.
        """
        return len(self.datapoints)
    
    def __getitem__(self, index: int) -> FlinkDatapoint:
        """
        Returns the datapoint at the given index.
        """
        return self.datapoints[index]

    def _load_datapoints(self) -> List[FlinkDatapoint]:
        """
        Loads all datapoints and their associated data.

        Returns:
            List[FlinkDatapoint]: The loaded FlinkDatapoint object.

        Raises:
            FileNotFoundError: If any of the required files are not found.
            ValueError: If the image or depth file cannot be opened.
        """

        # Load image, depth, label and metadata
        image_folder_path = self.base_path / "images"
        depth_folder_path = self.base_path / "depth"
        label_folder_path = self.base_path / "labels"
        metadata_folder_path = self.base_path / "metadata"

        datapoints = []

        if not all([image_folder_path.exists(), depth_folder_path.exists(), label_folder_path.exists(), metadata_folder_path.exists()]):
            raise FileNotFoundError(
                f"Required folders not found in {self.base_path}")

        for image_file in image_folder_path.iterdir():
            camid_timestamp = image_file.stem
            if all([
                image_file.exists(),
                (depth_folder_path / f"{camid_timestamp}.png").exists(),
                (label_folder_path / f"{camid_timestamp}.json").exists(),
                (metadata_folder_path / f"{camid_timestamp}.json").exists()
            ]):
                datapoints.append(FlinkDatapoint(
                    image=image_file,
                    depth=depth_folder_path / f"{camid_timestamp}.png",
                    label_data=LabelData.model_validate_json(
                        (label_folder_path /
                         f"{camid_timestamp}.json").read_text()
                    ),
                    metadata=Metadata.model_validate_json(
                        (metadata_folder_path /
                         f"{camid_timestamp}.json").read_text()
                    )
                ))

        return datapoints


# Example Usage
if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load a Flink dataset')
    parser.add_argument('dataset_path', type=str,
                        help='Path to the dataset directory')
    args = parser.parse_args()

    # Convert string path to Path object
    dataset_path = Path(args.dataset_path)

    # Initialize the dataset loader
    loader = FlinkDatasetLoader(dataset_path)

    print(len(loader))
    try:
        import matplotlib.pyplot as plt
        img = loader.datapoints[0].get_image()
        depth = loader.datapoints[0].get_depth()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(img)
        ax1.set_title('RGB Image')
        # Using plasma colormap for better depth visualization
        ax2.imshow(depth, cmap='plasma')
        ax2.set_title('Depth Image')
        plt.show()
    except Exception as e:
        print(e)
        print("Error: Matplotlib is not installed. Please install it using 'pip install matplotlib'.")
