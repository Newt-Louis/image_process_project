import json, torch, matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.transforms import functional as tvF
from pycocotools.coco import COCO

DEV = 'cuda'
IMG_DIR = '/content/gold_dataset/images/test'
coco = COCO('/content/gold_dataset/coco_annotations/instances_test.json')

# --- 1. Chọn 2-3 ảnh khó nhất: nhiều vật thể nhất ---
counts = {iid: len(coco.getAnnIds(imgIds=iid)) for iid in coco.imgs}
hard_ids = sorted(counts, key=counts.get, reverse=True)[:3]

# --- 2. Nạp 3 model ---
yolo = YOLO('/content/drive/MyDrive/UIT/graduate_project/checkpoints/yolov11/yolov11_product/weights/best.pt')

def load_tv(builder, ckpt):
    m = builder(weights=None, num_classes=2)
    m.load_state_dict(torch.load(ckpt, map_location=DEV)['model'])
    return m.to(DEV).eval()

frcnn = load_tv(fasterrcnn_resnet50_fpn,
    '/content/drive/MyDrive/UIT/graduate_project/checkpoints/fasterrcnn/best.pth')
retina = load_tv(retinanet_resnet50_fpn,
    '/content/drive/MyDrive/UIT/graduate_project/checkpoints/retinanet/best.pth')

CONF = 0.5  # ngưỡng hiển thị thống nhất cho cả 3 model

def draw(ax, img, boxes, color, title):
    ax.imshow(img); ax.set_title(title, fontsize=11); ax.axis('off')
    for x1, y1, x2, y2 in boxes:
        ax.add_patch(mpatches.Rectangle((x1, y1), x2-x1, y2-y1,
                     fill=False, edgecolor=color, linewidth=1.6))

fig, axes = plt.subplots(len(hard_ids), 4, figsize=(18, 4.6*len(hard_ids)))
if len(hard_ids) == 1: axes = axes[None, :]

for r, iid in enumerate(hard_ids):
    info = coco.imgs[iid]
    img = Image.open(f"{IMG_DIR}/{info['file_name']}").convert('RGB')

    # GT: coco [x,y,w,h] -> [x1,y1,x2,y2]
    gt = [[a['bbox'][0], a['bbox'][1], a['bbox'][0]+a['bbox'][2], a['bbox'][1]+a['bbox'][3]]
          for a in coco.loadAnns(coco.getAnnIds(imgIds=iid))]

    yb = yolo(img, conf=CONF, verbose=False)[0].boxes.xyxy.cpu().numpy()

    t = tvF.to_tensor(img).to(DEV)
    with torch.no_grad():
        fo, ro = frcnn([t])[0], retina([t])[0]
    fb = fo['boxes'][fo['scores'] >= CONF].cpu().numpy()
    rb = ro['boxes'][ro['scores'] >= CONF].cpu().numpy()

    n = counts[iid]
    draw(axes[r,0], img, gt, 'lime',    f'Ground-truth ({n} vật)')
    draw(axes[r,1], img, yb, 'dodgerblue', f'YOLOv11 ({len(yb)} khung)')
    draw(axes[r,2], img, fb, 'green',   f'Faster R-CNN ({len(fb)} khung)')
    draw(axes[r,3], img, rb, 'crimson', f'RetinaNet ({len(rb)} khung)')

plt.tight_layout()
plt.savefig('qualitative_comparison.png', dpi=150, bbox_inches='tight')