import math
import os.path as osp
from enum import Enum, auto


EXAMPLE_TEXTS = [
    "The University of Copenhagen (Danish: Københavns Universitet, abbr. KU) is a public research university in Copenhagen, Denmark. Founded in 1479, the University of Copenhagen is the second-oldest university in Scandinavia, after Uppsala University. The University of Copenhagen consists of six different faculties, with teaching taking place in its four distinct campuses, all situated in Copenhagen.[13][14] The university operates 36 different departments and 122 separate research centres in Copenhagen, as well as a number of museums and botanical gardens in and outside the Danish capital.[15] The University of Copenhagen also owns and operates multiple research stations around Denmark, with two additional ones located in Greenland.",
    "Originally a Viking fishing village established in the 10th century in the vicinity of what is now Gammel Strand, Copenhagen became the capital of Denmark in the early 15th century. During the 16th century, the city served as the de facto capital of the Kalmar Union and the seat of the Union's monarchy, which governed most of the modern-day Nordic region as part of a Danish confederation with Sweden and Norway. The city flourished as the cultural and economic centre of Scandinavia during the Renaissance.",
    "哥本哈根大学（丹麦语：Københavns Universitet），位于丹麦王国的首都哥本哈根，是丹麦历史最悠久的大学，也是规模第二大的大学之一。在读学生总数约4万人，超过半数为女性。此外，还有逾一万教职员工。哥本哈根大学的校园散落在市区里和城市周边，最古老的部分则位于哥本哈根古城区。哥本哈根大学是斯堪的纳维亚地区第二古老的大学，它和位于日德兰半岛的奥胡斯大学同为丹麦享有国际声誉的教育与科研机构。"
    "厄勒海峽大橋在2000年完工後，哥本哈根與瑞典的馬爾默可透過車輛和鐵路往來，促成了兩地人力資源的互相交流，每年利用厄勒海峽大橋的通勤人數不斷增長。此外大橋通車後也讓兩座城市之間形成北歐地區最大的城市群。",
    "कोपनहेगन विश्वविद्यालय (डेनिश: Københavns Universitet) डेनमार्क की शोधसंस्थान है। यह डेनमार्क की सबसे प्राचीन और दूसरा सबसे बड़ा विश्वविद्यालय है। इसकी स्थापना १४७९ में एक सामन्य अध्ययन क्षेत्र के रूप में हुई। यह स्कैंडेनेविया का द्वितीय सबसे पुराना विश्वविद्यालय है जिसकी स्थापना उप्साला विश्वविद्यालय (१४७७) के बाद हुई। विश्वविद्यालय में ३७००० से अधिक छात्र-छात्राएँ अध्ययनरत हैं और ७००० से अधिक कर्मचारी कार्यरत हैं।",
    "इस क्षेत्र के प्रथम लिखित दस्तावेज ११वीं सदी के हैं और कोपनहेगन १५वीं सदी के आरम्भ में और क्रिस्चियन चतुर्थ के शासनकाल में डेनमार्क की राजधानी बना। वर्ष २००० में ओरेसण्ड सेतु के पूरा होने के साथ ही कोपनहेगन ओरेसण्ड क्षेत्र का केन्द्र बन गया है। इस क्षेत्र में, कोपनहेगन और स्वीडन का माल्मो नगर मिलकर एक आम महानगरीय क्षेत्र बनने की प्रकिया में है। ५० किमी के अर्धव्यास में २७ लाख लोगों के साथ, कोपनहेगन उत्तरी यूरोप के सबसे सघन क्षेत्रों में से एक है। नॉर्डिक देशों में कोपनहेगन सर्वाधिक पधारा जाने वाला देश है जहाँ पर २००७ में १३ लाख विदेशी पर्यटक आए। ",
    "Реформація в 1536 році означала радикальну зміну в статусі й ролі університету в данському суспільстві. Але з організаційного погляду, університет мав залишатися навчальним закладом за середньовічною моделлю, яка визначала діяльність університету далеко на майбутнє. І тільки в 1771 році університет втратив свою власну юрисдикцію. А в другій половині XX століття, нарешті, остаточно зникає те, що називали «професорською владою». ",
    "З початку XXI століття Копенгаген зазнав потужний міський і культурний розвиток, якому сприяли інвестиції в його установи та інфраструктуру. Місто є культурним , економічним і урядовим центром Данії; це один із головних фінансових центрів Північної Європи з Копенгагенською фондовою біржею. Економіка Копенгагена швидко розвивалася у галузі послуг, особливо завдяки ініціативам у галузі інформаційних технологій, фармацевтичних препаратів і чистих технологій. Після завершення будівництва Ересуннського мосту Копенгаген дедалі більше інтегрувався зі шведською провінцією Сконе та її найбільшим містом Мальме, утворюючи Ересуннський регіон. Кілька мостів, що сполучають різні райони, міський пейзаж характеризується парками та набережними. Пам’ятки Копенгагена, такі як сад Тіволі, статуя Русалоньки, палаци Амалієнборг і Крістіансборг, замок Розенборг, церква Фредеріка, Берсен і багато музеїв, ресторанів і нічних клубів є важливими туристичними об’єктами. ",
]


class Modality(Enum):
    IMAGE = auto()
    TEXT = auto()


def get_root_dir() -> str:
    return osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))


def get_data_dir() -> str:
    return osp.join(get_root_dir(), 'data')


def get_log_dir() -> str:
    return osp.join(get_root_dir(), 'logs')


def get_fallback_fonts_dir() -> str:
    return osp.join(get_root_dir(), 'fonts')

