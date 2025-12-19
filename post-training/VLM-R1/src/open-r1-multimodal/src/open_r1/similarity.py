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

    def score(self, img1: Image.Image, img2: Image.Image) -> float:
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
    img1, img2 = process_imgs(img1, img2, 1024)
    # max_mae = np.mean(np.maximum(img1, 255 - img1))
    mae = np.mean(np.abs(img1 - img2))
    # return {"mae": mae, "normalized_mae": 1 - mae / max_mae}
    return mae

def clip_mae(img1, img2):
    """clip - mae/255"""
    mae = mae_score(img1, img2)
    clip = clip_scorer.score(img1, img2)
    return clip - mae/255 # scale mae by 255/200


import re
import base64

with open(os.getenv("CSS_PATH"), "r", encoding="utf-8") as f:
        TAILWIND_CSS = f.read()
with open(os.getenv("PLACEHOLDER_PATH"), "rb") as f:
        PLACEHOLDER = base64.b64encode(f.read()).decode("utf-8")

def preprocess_html(html_str: str) -> str:
    # 1. Load and wrap Tailwind CSS in <style>
    style_tag = f"<style>{TAILWIND_CSS}</style>"
    html_str = html_str.replace('<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">', style_tag)

    # 3. Convert placeholder image to base64 and replace all occurrences
    base64_url = f"data:image/png;base64,{PLACEHOLDER}"
    html_str = html_str.replace("placeholder.jpg", base64_url)

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


def rendered_score(gen_html, ref_html, score_func, verbose=True):
    """Calculate visual similarity score between two HTML documents using screenshots."""

    # if not verbose:
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as ref_file, \
    #         tempfile.NamedTemporaryFile(delete=False, suffix=".png") as gen_file:

    #         ref_path = ref_file.name
    #         gen_path = gen_file.name

    ref_name = str(abs(hash(ref_html)))[:6]
    gen_name = f"{ref_name}_{str(abs(hash(gen_html)))[:6]}"
    ref_path = f"./images/{ref_name}.png"
    gen_path = f"./images/{gen_name}.png"
    # with open(f"./images/{gen_name}.html", "w") as f:
    #     f.write(gen_html)


    try:
        # Generate screenshots synchronously
        generate_screenshot(ref_html, ref_path)
        generate_screenshot(gen_html, gen_path)

        # Calculate similarity score
        with Image.open(ref_path) as ref_img, Image.open(gen_path) as gen_img:
            if type(score_func) == list:
                score = []
                for func in score_func:
                    score.append(func(ref_img, gen_img))
                if not verbose:
                    os.remove(ref_path)
                    os.remove(gen_path)
                return np.mean(score)
                
            if not verbose:
                os.remove(ref_path)
                os.remove(gen_path)
            return score_func(ref_img, gen_img)
            
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

def process_item(i, gt, base_json, sft_json, clip_scorer):
    gt_html = gt[i]
    base_html = base_json[i]
    sft_html = sft_json[i]
    
    # calculate score
    base = rendered_score(gt_html, base_html, [ mae_score, clip_scorer.score])
    sft = rendered_score(gt_html, sft_html, [ mae_score, clip_scorer.score])
    
    return {
        # "emd_base": emd_base,
        # "emd_sft": emd_sft,
        "base": base,
        "sft": sft,
    }

if __name__ == "__main__":
    # Load data
    # gt = json.load(open("test_selected.json"))
    # gt = [x["conversations"][1]["value"] for x in gt]
    
    # # Load jsonl files
    # with open("mrweb_3b_original.jsonl", "r") as f:
    #     base_json = [json.loads(line)["predict"] for line in f.readlines()]
    # with open("mrweb_3b_sft_2000.jsonl", "r") as f:
    #     sft_json = [json.loads(line)["predict"] for line in f.readlines()]

    # # Initialize CLIP scorer (once per process)
    # clip_scorer = CLIPScorer()

    # # Create multiprocessing pool
    # from multiprocessing.pool import ThreadPool
    # num_processes = 6
    # with ThreadPool(processes=num_processes) as pool:
    #     func = partial(process_item, gt=gt, base_json=base_json, 
    #                   sft_json=sft_json, clip_scorer=clip_scorer)
    #     results = list(tqdm.tqdm(
    #         pool.imap(func, range(len(gt))),
    #         total=len(gt)))

    # # Summarize results
    # base_scores = [x["base"] for x in results]
    # sft_scores = [x["sft"] for x in results]
    # print("base scores: ", np.mean(base_scores))

    # write a minimal code to test playwright screenshot
    html = """<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Interior Design Firm</title>
    <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
</head>
<body class="bg-gray-100 font-sans leading-normal max-w-3xl mx-auto">
    <section class="flex flex-col items-center justify-between p-6">
        <section class="flex-1 text-left">
            <h1 class="text-3xl font-bold mb-4 text-gray-800">Interior Design Firm</h1>
            <p class="mb-4 text-gray-600">Welcome to our interior design firm, where we specialize in creating beautiful and functional spaces that reflect your individual style and personality. Our team of experienced designers work closely with you to understand your needs and vision, and then transform that into a reality. We believe that every space is a reflection of the people who inhabit it, and we strive to make that connection as strong as possible.</p>
            <h2 class="text-2xl font-bold mb-2">Testimonials</h2>
            <p class="mb-4 text-gray-600">"I was blown away by the professionalism and creativity of the interior design team at our new home. They truly understood my vision and brought it to life. I couldn't be happier with the results." - John D.</p>
            <p class="mb-4 text-gray-600">"I was hesitant to hire a professional designer, but I'm so glad I did. The results are beyond my expectations. I'm so happy with the space I've created for myself and my family." - Jane S.</p>
        </section>
        <section class="container mx-auto ml-72 mt-4">
            <img src="placeholder.jpg" class="w-1/3 mx-auto h-48 rounded-full shadow-lg" alt="Testimonial image">
        </section>
    </section>
</body>
</html>"""

    path1 = "test1.png"
    generate_screenshot(html, path1)
    # print(rendered_score(html_ref, html_gen, [mae_score, CLIPScorer().score]))
    # generate_screenshot(html_ref, "test.png")
