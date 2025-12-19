import tempfile
import os
from PIL import Image
from playwright.sync_api import sync_playwright
import os
import open_clip
import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from lap import lapjv
from multiprocessing import Pool
# from selenium import webdriver
# from selenium.webdriver.firefox.options import Options
os.makedirs("./images", exist_ok=True)

def solve_assignment_lapjv(cost_matrix):
    _, col_idx, _ = lapjv(cost_matrix)
    return col_idx


def process_imgs(image1, image2, max_size):
    # Get the original sizes
    width1, height1 = image1.size
    width2, height2 = image2.size

    # Determine the new dimensions (max of both images' width and height)
    new_width = max(width1, width2)
    new_height = max(height1, height2)

    # Pad images to the new dimensions with random values
    def pad_image(image, new_width, new_height):
        # Create a random padded background with the new dimensions
        random_padding = np.random.randint(0, 256, (new_height, new_width, 3), dtype=np.uint8)
        padded_image = Image.fromarray(random_padding)

        # Paste the original image onto the padded background (placing in the top-left corner)
        padded_image.paste(image, (0, 0))

        return padded_image

    padded_image1 = pad_image(image1, new_width, new_height)
    padded_image2 = pad_image(image2, new_width, new_height)

    # Calculate the aspect ratio for resizing to the max size
    aspect_ratio = min(max_size / new_width, max_size / new_height)
    new_size = (int(new_width * aspect_ratio), int(new_height * aspect_ratio))

    # Resize the padded images to the specified max size
    resized_image1 = padded_image1.resize(new_size, Image.LANCZOS)
    resized_image2 = padded_image2.resize(new_size, Image.LANCZOS)

    # resized_image1.show()
    # resized_image2.show()

    # Convert the images to numpy arrays with dtype int16
    array1 = np.array(resized_image1).astype(np.int16)
    array2 = np.array(resized_image2).astype(np.int16)

    return array1, array2



def calculate_emd_sim(img_array1, img_array2):
    """img_array1 is the original image, img_array2 is the generated image"""
    if len(img_array1.shape) == 2:
        flat_array1 = img_array1.flatten()
        flat_array2 = img_array2.flatten()

        cost_matrix = np.abs(flat_array1[:, None] - flat_array2[None, :])
        _, col_idx, _ = lapjv(cost_matrix)

        total_min_cost = cost_matrix[np.arange(len(flat_array1)), col_idx].sum()
        max_cost = np.maximum(flat_array1, 255 - flat_array1).sum()
        normalized_min_cost = total_min_cost / max_cost

    else:
        red1, green1, blue1 = img_array1[:, :, 0], img_array1[:, :, 1], img_array1[:, :, 2]
        red2, green2, blue2 = img_array2[:, :, 0], img_array2[:, :, 1], img_array2[:, :, 2]

        flat_red1, flat_green1, flat_blue1 = red1.flatten(), green1.flatten(), blue1.flatten()
        flat_red2, flat_green2, flat_blue2 = red2.flatten(), green2.flatten(), blue2.flatten()

        cost_matrix_red = np.abs(flat_red1[:, None] - flat_red2[None, :]).astype(np.float32)
        cost_matrix_green = np.abs(flat_green1[:, None] - flat_green2[None, :]).astype(np.float32)
        cost_matrix_blue = np.abs(flat_blue1[:, None] - flat_blue2[None, :]).astype(np.float32)

        with Pool(processes=3) as pool:
            results = pool.map(solve_assignment_lapjv, [cost_matrix_red, cost_matrix_green, cost_matrix_blue])
            col_ind_red = results[0]
            col_ind_green = results[1]
            col_ind_blue = results[2]
        
        min_cost_red_lapjv = cost_matrix_red[np.arange(len(flat_red1)), col_ind_red].sum()
        min_cost_green_lapjv = cost_matrix_green[np.arange(len(flat_green1)), col_ind_green].sum()
        min_cost_blue_lapjv = cost_matrix_blue[np.arange(len(flat_blue1)), col_ind_blue].sum()

        total_min_cost_lapjv = min_cost_red_lapjv + min_cost_green_lapjv + min_cost_blue_lapjv
        max_cost = np.maximum(flat_red1, 255 - flat_red1).sum() + np.maximum(flat_green1, 255 - flat_green1).sum() + np.maximum(flat_blue1, 255 - flat_blue1).sum()
        normalized_min_cost = total_min_cost_lapjv / max_cost

    # return {"cost": total_min_cost_lapjv, "normalized_sim": 1 - normalized_min_cost}
    return 1 - normalized_min_cost

def emd_similarity(image1_path, image2_path, max_size=64, mode="L"):
    """not symmetric, the first image is the original image, the score is normalized according to the original image"""
    image1 = Image.open(image1_path).convert(mode) if type(image1_path) == str else image1_path.convert(mode)
    image2 = Image.open(image2_path).convert(mode) if type(image2_path) == str else image2_path.convert(mode)

    array1, array2 = process_imgs(image1, image2, max_size)
    similarity = calculate_emd_sim(array1, array2)

    return similarity

class CLIPScorer:
    def __init__(self, model_name='ViT-B-32-quickgelu', pretrained='openai'):
        """
        Initializes the CLIPScorer with the specified model.

        Args:
            model_name (str): The name of the CLIP model to use.
            pretrained (str): Specifies whether to load pre-trained weights.
        """
        self.device = "cuda" if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
        self.device = "cpu"  # Force CPU for compatibility
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)

    def clip_score(self, img1: Image.Image, img2: Image.Image) -> float:
        """
        Calculates the CLIP score (cosine similarity) between two images.

        Args:
            img1 (Image.Image): The first image as a PIL Image.
            img2 (Image.Image): The second image as a PIL Image.

        Returns:
            float: The cosine similarity score between the two images.
        """
        # Preprocess the images
        image1 = self.preprocess(img1).unsqueeze(0).to(self.device)
        image2 = self.preprocess(img2).unsqueeze(0).to(self.device)

        # Get the image features from CLIP using openclip
        with torch.no_grad():
            image1_features = self.model.encode_image(image1)
            image2_features = self.model.encode_image(image2)

        # Normalize the features to unit length
        image1_features /= image1_features.norm(dim=-1, keepdim=True)
        image2_features /= image2_features.norm(dim=-1, keepdim=True)

        # Calculate cosine similarity between the two image features
        cosine_similarity = torch.nn.functional.cosine_similarity(image1_features, image2_features)
        return cosine_similarity.item()

clip_scorer = CLIPScorer() 

def ssim_score(img1, img2):
    # resize images to match the size of the smaller image
    img1, img2 = process_imgs(img1, img2, 512)
    return ssim(img1, img2, channel_axis=-1, data_range=255)


def mae_score(img1, img2):
    """mean absolute error, it is a pixel-based metric"""
    img1, img2 = process_imgs(img1, img2, 512)
    # max_mae = np.mean(np.maximum(img1, 255 - img1))
    mae = np.mean(np.abs(img1 - img2))
    # return {"mae": mae, "normalized_mae": 1 - mae / max_mae}
    return mae

def clip_mae(img1, img2):
    """clip - mae/255"""
    mae = mae_score(img1, img2)
    clip = clip_scorer.clip_score(img1, img2)
    return clip - mae/255

CSS_PATH="/apdcephfs_qy3/share_301812049/jarviswang/wt/yx/VLM-R1/src/open-r1-multimodal/src/open_r1/tailwind.min.css"
PLACEHOLDER_PATH="/apdcephfs_qy3/share_301812049/jarviswang/wt/yx/VLM-R1/src/open-r1-multimodal/src/open_r1/placeholder.jpg"

def preprocess_html(html_str: str) -> str:
    # css_path = os.path.join(os.path.dirname(__file__), "tailwind.min.css")
    # placeholder_path = os.path.join(os.path.dirname(__file__), "placeholder.jpg")
    html_str = html_str.replace("placeholder.jpg", PLACEHOLDER_PATH)
    html_str = html_str.replace("https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css", CSS_PATH)
    return html_str


def generate_screenshot(html_content, path):

    html_content = preprocess_html(html_content)

    with sync_playwright() as p:
        
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Set consistent rendering parameters
        page.set_viewport_size({"width": 1280, "height": 720})
        page.route("**/*", lambda route: route.continue_())  # Allow external resources

        
        # Render and screenshot
        page.set_content(html_content, timeout=100000)
        page.wait_for_load_state("networkidle", timeout=100000)
        page.screenshot(
            path=path,
            full_page=True,
            type="png",
        )
        browser.close()


def rendered_score(ref_html, gen_html, score_func, verbose=True):
    """Calculate visual similarity score between two HTML documents using screenshots."""

    if not verbose:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ref_file, \
            tempfile.NamedTemporaryFile(delete=False, suffix=".png") as gen_file:

            ref_path = ref_file.name
            gen_path = gen_file.name
    else:
        ref_name = str(abs(hash(ref_html)))[:6]
        gen_name = f"{ref_name}_{str(abs(hash(gen_html)))[:6]}"
        ref_path = f"./images/{ref_name}.png"
        gen_path = f"./images/{gen_name}.png"
        with open(f"./images/{gen_name}.html", "w") as f:
            f.write(gen_html)


    try:
        # Generate screenshots synchronously
        generate_screenshot(ref_html, ref_path)
        generate_screenshot(gen_html, gen_path)

        # Calculate similarity score
        with Image.open(ref_path) as ref_img, Image.open(gen_path) as gen_img:
            if type(score_func) == list:
                score = {}
                for func in score_func:
                    score.update({func.__name__: func(ref_img, gen_img)})
                return score

            return {score_func.__name__: score_func(ref_img, gen_img)}
            
    finally:
        # Cleanup temp files
        # os.unlink(ref_path)
        # os.unlink(gen_path)
        pass



import json
import tqdm
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial

def process_item(i, gt, exp_json, clip_scorer):
    gt_html = gt[i]
    exp_html = exp_json[i]
    
    # calculate score
    score = rendered_score(gt_html, exp_html, [mae_score, clip_scorer.clip_score])
    return score

if __name__ == "__main__":
    # test_path = "../LLaMA-Factory/data/CodeMLLM/websight/test.json"
    test_path="./test_selected.json"
    exp_name = "original"
    exp_path = "mrweb_3b_original.jsonl"
    # Load data
    gt = json.load(open(test_path, "r"))
    gt = [x["conversations"][1]["value"] for x in gt]


    with open(exp_path, "r") as f:
        exp_json = [json.loads(line)["predict"] for line in f.readlines()]
    
    
    # # Load jsonl files
    # with open("mrweb_3b_original.jsonl", "r") as f:
    #     base_json = [json.loads(line)["predict"] for line in f.readlines()]
    # with open("mrweb_3b_sft_2000.jsonl", "r") as f:
    #     sft_json = [json.loads(line)["predict"] for line in f.readlines()]

    # Initialize CLIP scorer (once per process)
    clip_scorer = CLIPScorer()

    # Create multiprocessing pool
    from multiprocessing.pool import ThreadPool
    num_processes = 6
    with ThreadPool(processes=num_processes) as pool:
        func = partial(process_item, gt=gt, exp_json=exp_json, clip_scorer=clip_scorer)
        results = list(tqdm.tqdm(
            pool.imap(func, range(len(gt))),
            total=len(gt)))

    # Summarize average results 
    keys = results[0].keys()
    summary = {key: [] for key in keys}
    for result in results:
        for key in keys:
            summary[key].append(result[key])
    summary = {key: np.mean(value) for key, value in summary.items()}
    print(f"Average results for {exp_name}:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


    