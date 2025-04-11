from PIL import Image, ImageDraw, ImageFont


def add_header_to_image(img: Image, title: str, subtitle: str) -> Image:
    """Add page title and URL header to screenshot"""
    width, height = img.size

    font_title_size = 22
    font_url_size = 18

    possible_fonts = [
        "arial.ttf",
        "Helvetica.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf"
    ]

    font_title = None
    font_url = None

    for font_path in possible_fonts:
        try:
            font_title = ImageFont.truetype(font_path, size=font_title_size)
            font_url = ImageFont.truetype(font_path, size=font_url_size)
            break
        except IOError:
            continue

    if font_title is None or font_url is None:
        font_title = ImageFont.load_default()
        font_url = ImageFont.load_default()

    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)

    title_bbox = draw.textbbox((0, 0), title, font=font_title)
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=font_url)

    title_width = title_bbox[2] - title_bbox[0]
    title_height = title_bbox[3] - title_bbox[1]
    url_width = subtitle_bbox[2] - subtitle_bbox[0]
    url_height = subtitle_bbox[3] - subtitle_bbox[1]

    header_height = title_height + url_height + 50
    header = Image.new('RGB', (width, header_height), color='white')
    draw = ImageDraw.Draw(header)

    title_x = (width - title_width) / 2
    title_y = 15
    subtitle_x = (width - url_width) / 2
    subtitle_y = title_y + title_height + 10

    draw.text((title_x, title_y), title, font=font_title, fill='black')
    draw.text((subtitle_x, subtitle_y), subtitle, font=font_url, fill='gray')
    draw.line([(0, header_height - 1), (width, header_height - 1)], fill='black', width=1)

    combined = Image.new('RGB', (width, header_height + height), color='white')
    combined.paste(header, (0, 0))
    combined.paste(img, (0, header_height))

    return combined