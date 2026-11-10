from font_download.fonts import FontSource
from font_download.example_fonts.noto_sans import FONTS_NOTO_SANS

# Same google/fonts commit that FONTS_NOTO_SANS is pinned to.
_COMMIT = "601944d2ee97e76d4c456dc3d790db358c0c6ec5"

# Google Fonts discontinued Noto Sans Tibetan; Noto Serif Tibetan is the only
# Noto font covering the Tibetan block (bod_Tibt, dzo_Tibt) at this commit.
FONT_NOTO_SERIF_TIBETAN = FontSource(
    url=f"https://github.com/google/fonts/raw/{_COMMIT}/ofl/notoseriftibetan/NotoSerifTibetan%5Bwght%5D.ttf",
    name="NotoSerifTibetan[wght].ttf",
)

FONTS_NOTO_SANS_EXTENDED = FONTS_NOTO_SANS + [FONT_NOTO_SERIF_TIBETAN]
