from pycocotools.ytvos import YTVOS
from pycocotools._mask import frUncompressedRLE
import pycocotools.mask as mask_util
import numpy as np
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog,MetadataCatalog
import random
import cv2
import os
from skimage import measure
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


class YoutubeVos():
    CLASSES = ['place_holder','person', 'giant_panda', 'lizard', 'parrot', 'skateboard', 'sedan',
               'ape', 'dog', 'snake', 'monkey', 'hand', 'rabbit', 'duck', 'cat', 'cow', 'fish',
               'train', 'horse', 'turtle', 'bear', 'motorbike', 'giraffe', 'leopard',
               'fox', 'deer', 'owl', 'surfboard', 'airplane', 'truck', 'zebra', 'tiger',
               'elephant', 'snowboard', 'boat', 'shark', 'mouse', 'frog', 'eagle', 'earless_seal',
               'tennis_racket']

    ## videos->frames->bbox
    ## get video id -> get frame id -> attach annotations to that frame
    ## for example, one video has 20 images. and two sets of annotations for those images.
    ## attach those annotation to each frame.
    dataset_dict=[]
    def __init__(self,
                 ann_file,
                 img_prefix,
                 test_mode=False
                 ):
        self.img_prefix = img_prefix
        self.vid_infos = self.load_annotations(ann_file,img_prefix)
        # check_valid_videos(self.vid_infos)
        img_ids = []
        for idx, vid_info in enumerate(self.vid_infos):
            for frame_id in range(len(vid_info["filenames"])):
                img_ids.append((idx,frame_id))
        self.img_ids = img_ids
        # if not test_mode:
        #     valid_inds = [i for i, (v, f) in enumerate(self.img_ids)
        #                   if len(self.get_ann_info(v, f)['bboxes'])]
        #     self.img_ids = [self.img_ids[i] for i in valid_inds]
        for frame in self.img_ids:
            self.get_coco_annotations(frame[0],frame[1])

    # def check_valid_videos(self,):
    def get_coco_annotations(self,id,f_id):
        # loadvids(1)
        # 'width'
        # 'height'
        # 'length'
        # 'id'
        # 'filenames'[0] = '0043f083b5/'

        frame_record={}
        frame_record["video_file_name"] = self.vid_infos[id]['filenames'][0].split("/")[0]
        frame_record["file_name"] = self.img_prefix+self.vid_infos[id]['filenames'][f_id]
        frame_record["width"]= self.vid_infos[id]['width']
        frame_record["height"] = self.vid_infos[id]['height']
        frame_record["video_id"] = self.vid_infos[id]['id']
        frame_record["num_of_frames"]=self.vid_infos[id]['length']
        frame_record["image_id"] = self.vid_infos[id]['filenames'][f_id].split("/")[1]
        # ....
        video_id = frame_record["video_id"]
        ann_ids = self.ytvos.getAnnIds(vidIds=[video_id])
        ann_info = self.ytvos.loadAnns(ann_ids)
        ##bbox,bbox_mode=xyxy_abs,category_id:int,segmentation:[list],is_crowd
        objs= []
        for i, ann in enumerate(ann_info):
            if ann['bboxes'][f_id] is None:
                continue
            obj = {}
            bbox = ann['bboxes'][f_id]
            area = ann['areas'][f_id]
            # polygon-mask
            # binaryMask = self.ytvos.annToMask(ann,f_id)
            # segm = self.binary_mask_to_polygon(binaryMask)

            # bitmask
            # pycocotools.mask.frPyObjects(rle, rle['size'][0], rle['size'][1])
            uncom_bitmask = ann["segmentations"][f_id]
            # frUncompressedRLE([pyobj], h, w)[0]
            bitRLE= frUncompressedRLE([uncom_bitmask], uncom_bitmask['size'][0], uncom_bitmask['size'][1])
            # bitmask = bitRLE
            cat_id = ann['category_id']
            x1, y1, w, h = bbox
            if area <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            obj["bbox"] = bbox
            obj["bbox_mode"] = BoxMode.XYXY_ABS
            obj["segmentation"] = bitRLE[0]
            obj["category_id"] = cat_id
            obj["area"] = area
            obj["iscrowd"] = ann['iscrowd']
            objs.append(obj)

        if(objs):
            frame_record["annotations"] = objs
            self.dataset_dict.append(frame_record)
        pass
        # if not test_mode:
        #     valid_inds = [ i for i, (v,f) in enumerate (self.img_ids)
        #                    if len(self.get_ann_info(v,f)['bboxes'])]
        #     self.img_ids = [self.img_ids [i] for i in valid_inds ]
        # transforms
        # self.img_transform = ImageTransform(
        #     size_divisor=self.size_divisor, **self.img_norm_cfg)
        # self.bbox_transform = BboxTransform()
        # self.mask_transform = MaskTransform()
        # self.numpy2tensor = Numpy2Tensor()

    def close_contour(self,contour):
        if not np.array_equal(contour[0], contour[-1]):
            contour = np.vstack((contour, contour[0]))
        return contour

    def binary_mask_to_polygon(self,binary_mask, tolerance=0):
        """Converts a binary mask to COCO polygon representation
        Args:
            binary_mask: a 2D binary numpy array where '1's represent the object
            tolerance: Maximum distance from original points of polygon to approximated
                polygonal chain. If tolerance is 0, the original coordinate array is returned.
        """
        polygons = []
        # pad mask to close contours of shapes which start and end at an edge
        padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
        contours = measure.find_contours(padded_binary_mask, 0.5)
        contours = np.subtract(contours, 1)
        for contour in contours:
            contour = self.close_contour(contour)
            contour = measure.approximate_polygon(contour, tolerance)
            if len(contour) < 3:
                continue
            contour = np.flip(contour, axis=1)
            segmentation = contour.ravel().tolist()
            # after padding and subtracting 1 we may get -0.5 points in our segmentation
            segmentation = [0 if i < 0 else i for i in segmentation]
            polygons.append(segmentation)

        return polygons

    def get_dataset_dict(self):
        return self.dataset_dict

    def load_annotations(self,ann_file,folder_path):
        self.ytvos = YTVOS(ann_file)
        self.cat_ids = self.ytvos.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.vid_ids = self.ytvos.getVidIds()
        vid_infos = []
        for i in self.vid_ids:
            info = self.ytvos.loadVids([i])[0]
            info_folder = folder_path+self.ytvos.loadVids([i])[0]['file_names'][0].split("/")[0]
            try:
                if(os.listdir(info_folder)):
                    info['filenames'] = info['file_names']
                    vid_infos.append(info)
                else:
                    print(info_folder)
                    continue
            except: continue
        return vid_infos

    def get_ann_info(self,idx, frame_id):
        vid_id = self.vid_infos[idx]['id']
        ann_ids = self.ytvos.getAnnIds(vidIds = [vid_id])
        ann_info  = self.ytvos.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info,frame_id)

    # def _parse_ann_info(self,ann_info,frame_id,with_mask=True):
    #     """Parse bbox and mask annotation.
    #
    #     Args:
    #         ann_info (list[dict]): Annotation info of an image.
    #         with_mask (bool): Whether to parse mask annotations.
    #
    #     Returns:
    #         dict: A dict containing the following keys: bboxes, bboxes_ignore,
    #             labels, masks, mask_polys, poly_lens.
    #     """
    #
    #     ##bbox,bbox_mode=xyxy_abs,category_id:int,segmentation:[list],is_crowd
    #     gt_bboxes = []
    #     gt_labels = []
    #     gt_ids = []
    #     gt_bboxes_ignore = []
    #     # Two formats are provided.
    #     # 1. mask: a binary map of the same size of the image.
    #     # 2. polys: each mask consists of one or several polys, each poly is a
    #     # list of float.
    #     if with_mask:
    #         gt_masks = []
    #         gt_mask_polys = []
    #         gt_poly_lens = []
    #     for i, ann in enumerate(ann_info):
    #         # each ann is a list of masks
    #         # ann:
    #         # bbox: list of bboxes
    #         # segmentation: list of segmentation
    #         # category_id
    #         # area: list of area
    #         bbox = ann['bboxes'][frame_id]
    #         area = ann['areas'][frame_id]
    #         segm = ann['segmentations'][frame_id]
    #         if bbox is None: continue
    #         else:
    #             print("line 98 ytvosdataset")
    #         x1, y1, w, h = bbox
    #         if area <= 0 or w < 1 or h < 1:
    #             continue
    #         bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
    #         if ann['iscrowd']:
    #             gt_bboxes_ignore.append(bbox)
    #         else:
    #             gt_bboxes.append(bbox)
    #             gt_ids.append(ann['id'])
    #             gt_labels.append(self.cat2label[ann['category_id']])
    #         if with_mask:
    #             gt_masks.append(self.ytvos.annToMask(ann, frame_id))
    #             mask_polys = [
    #                 p for p in segm if len(p) >= 6
    #             ]  # valid polygons have >= 3 points (6 coordinates)
    #             poly_lens = [len(p) for p in mask_polys]
    #             gt_mask_polys.append(mask_polys)
    #             gt_poly_lens.extend(poly_lens)
    #     if gt_bboxes:
    #         gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
    #         gt_labels = np.array(gt_labels, dtype=np.int64)
    #     else:
    #         gt_bboxes = np.zeros((0, 4), dtype=np.float32)
    #         gt_labels = np.array([], dtype=np.int64)
    #
    #     if gt_bboxes_ignore:
    #         gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
    #     else:
    #         gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
    #
    #     ann = dict(
    #         bboxes=gt_bboxes,
    #         labels=gt_labels,
    #         obj_ids=gt_ids,
    #         bboxes_ignore=gt_bboxes_ignore
    #     )
    #
    #     if with_mask:
    #         ann['masks'] = gt_masks
    #         # poly format is not used in the current implementation
    #         ann['mask_polys'] = gt_mask_polys
    #         ann['poly_lens'] = gt_poly_lens
    #     return ann

youtube_vis_annotations_file = "../../youtube-vis/strain_200.json"
youtube_vos_train_image = "../../../youtube-vos/train/JPEGImages/"
youtube_vis_test_file = "../../youtube-vis/stest_500.json"
youtube_vos_test_image = "../../../youtube-vos/train/JPEGImages/"

ytvos = YoutubeVos(youtube_vis_annotations_file,youtube_vos_train_image)
dataset_dicts=ytvos.dataset_dict
DatasetCatalog.register("youtube_vos",ytvos.get_dataset_dict)
MetadataCatalog.get("youtube_vos").set(thing_classes=ytvos.CLASSES)
video_metadata = MetadataCatalog.get("youtube_vos")

ytvos_test = YoutubeVos(youtube_vis_test_file,youtube_vos_test_image)
DatasetCatalog.register("youtube_vos_test",ytvos_test.get_dataset_dict)
MetadataCatalog.get("youtube_vos_test").set(thing_classes=ytvos_test.CLASSES)
dataset_dicts_test = ytvos_test.dataset_dict
video_metadata_test = MetadataCatalog.get("youtube_vos_test")

for d in random.sample(dataset_dicts, 3):
    total_frames=len(os.listdir(d['file_name'].split("/")[0]))
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=video_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    cv2.imshow("windnow",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyWindow("window")
cv2.destroyAllWindows()


# ytvos.load_annotations(youtube_vis_annotations_file)
# ytvos.get_ann_info()

### one annotation for one image -> should include
#https://github.com/facebookresearch/Detectron/issues/100



cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("youtube_vos",)
# cfg.DATASETS.TEST = ()
cfg.INPUT.MASK_FORMAT = "bitmask"
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
# cfg.MODEL.MASK_ON = True
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000   # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(ytvos.CLASSES)  # only has one class (ballon)
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
print(cfg)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=True)
trainer.train()

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
cfg.DATASETS.TEST = ("youtube_vos_test", )
predictor = DefaultPredictor(cfg)



for d in random.sample(dataset_dicts_test, 3):
    im = cv2.imread(youtube_vos_test_image+d["file_name"])
    outputs = predictor(im)
    print(d["file_name"])
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    # instances = outputs['instances']
    # instances.pred_masks_rle = [mask_util.encode(np.asfortranarray(mask)) for mask in instances.pred_masks]
    # for rle in instances.pred_masks_rle:
    #     rle['counts'] = rle['counts'].decode('utf-8')
    # print(instances.pred_masks_rle)
    v = Visualizer(im[:, :, ::-1],
                   metadata=video_metadata_test,
                   scale=0.8,
                   # instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("windows",out.get_image()[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyWindow("windows")



# instances = outputs['instances']
#
# instances.pred_masks_rle = [mask_util.encode(np.asfortranarray(mask)) for mask in instances.pred_masks]
# for rle in instances.pred_masks_rle:
#     rle['counts'] = rle['counts'].decode('utf-8')
#
# instances.remove('pred_masks')
#
# # TO TEST INVERT CONVERSION
# instances.pred_masks = np.stack([mask_util.decode(rle) for rle in instances.pred_masks_rle])


evaluator = COCOEvaluator("youtube_vos_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "youtube_vos_test")
print(inference_on_dataset(trainer.model, val_loader, evaluator))