import os
import numpy as np
import torch
import clip
import argparse
from dataset import ImageFolder

clip.available_models()


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class DummyArgs:
    pass


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


imagenet_classes = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray",
                    "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco",
                    "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper",
                    "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt",
                    "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog",
                    "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle",
                    "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama",
                    "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon",
                    "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake",
                    "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake",
                    "water snake", "vine snake", "night snake", "boa constrictor", "African rock python",
                    "Indian cobra", "green mamba", "sea snake", "Saharan horned viper",
                    "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion",
                    "yellow garden spider", "barn spider", "European garden spider", "southern black widow",
                    "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse",
                    "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw",
                    "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar",
                    "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus",
                    "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode",
                    "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab",
                    "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
                    "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret",
                    "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard",
                    "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican",
                    "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua",
                    "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon",
                    "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound",
                    "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound",
                    "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound",
                    "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner",
                    "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier",
                    "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier",
                    "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier",
                    "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier",
                    "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier",
                    "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier",
                    "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever",
                    "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer",
                    "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel",
                    "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel",
                    "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard",
                    "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie",
                    "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann",
                    "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog",
                    "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff",
                    "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky",
                    "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog",
                    "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon",
                    "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle",
                    "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf",
                    "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox",
                    "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat",
                    "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah",
                    "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat",
                    "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle",
                    "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect",
                    "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly",
                    "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly",
                    "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber",
                    "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot",
                    "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog",
                    "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep",
                    "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel",
                    "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo",
                    "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon",
                    "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey",
                    "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey",
                    "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant",
                    "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish",
                    "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion",
                    "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
                    "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack",
                    "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail",
                    "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball",
                    "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon",
                    "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass",
                    "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse",
                    "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap",
                    "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom",
                    "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron",
                    "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit",
                    "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player",
                    "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence",
                    "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet",
                    "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs",
                    "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock",
                    "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet",
                    "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed",
                    "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer",
                    "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth",
                    "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum",
                    "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive",
                    "entertainment center", "envelope", "espresso machine", "face powder", "feather boa",
                    "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair",
                    "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car",
                    "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump",
                    "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano",
                    "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray",
                    "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief",
                    "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater",
                    "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass",
                    "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw",
                    "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer",
                    "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine",
                    "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass",
                    "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit",
                    "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup",
                    "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can",
                    "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home",
                    "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap",
                    "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap",
                    "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier",
                    "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ",
                    "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle",
                    "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel",
                    "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone",
                    "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum",
                    "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow",
                    "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium",
                    "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van",
                    "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug",
                    "printer", "prison", "missile", "projector", "hockey puck", "punching bag", "purse", "quill",
                    "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel",
                    "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control",
                    "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball",
                    "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong",
                    "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor",
                    "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store",
                    "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap",
                    "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door",
                    "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock",
                    "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater",
                    "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight",
                    "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf",
                    "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine",
                    "suit", "sundial", "sunglasses", "sunglasses", "sunscreen", "suspension bridge", "mop",
                    "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank",
                    "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof",
                    "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop",
                    "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck",
                    "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone",
                    "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano",
                    "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine",
                    "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe",
                    "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower",
                    "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle",
                    "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt",
                    "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket",
                    "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette",
                    "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli",
                    "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
                    "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange",
                    "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate",
                    "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito",
                    "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser",
                    "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom",
                    "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip",
                    "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom",
                    "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

print(f"{len(imagenet_classes)} classes, {len(imagenet_templates)} templates")


def zeroshot_classifier(classnames, templates, clip_model):
    with torch.no_grad():
        zeroshot_weights = []
        text_prompts = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            text_prompts.append(texts)  # format with class
            texts = clip.tokenize(texts).cuda()  # tokenize
            class_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights, text_prompts


def evaluate_zs(model, zs_weights, dataloader):
    device = "cuda"
    logit_scale = 100.0
    top1 = AverageMeter("ACC@1", ":6.2f")
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ zs_weights).softmax(dim=-1)
            logits = similarity * logit_scale
            acc1 = accuracy(logits, labels, topk=(1,))
            top1.update(acc1[0].item(), len(labels))

    print(
        f"Zero-shot CLIP top-1 accuracy : {top1.avg:.2f}"
    )
    return top1.avg


def get_clip_model(name):
    clip_model, preprocess = clip.load(name)
    clip_model.eval()
    return clip_model, preprocess

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', default="output", type=str,
                        help='Where to save the adversarial examples, and other results')
    parser.add_argument('--data_path', default="path/to/dataset", type=str,
                        help='The clean images root directory')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for the dataloader')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_parser()

    model_names = ['RN50',
                   'RN101',
                   'RN50x4',
                   'RN50x16',
                   'ViT-B/32',
                   'ViT-B/16',
                   'ViT-L/14', ]


    data_path = args.data_path
    SAVE_DIR = args.save_dir
    os.makedirs(SAVE_DIR, exist_ok=True)



    accuracy_per_model = []

    for model_name in model_names:
        print(f"Model: {model_name}")
        clip_model, preprocess = get_clip_model(model_name)
        imagenet_dataset = ImageFolder(root=data_path,
                                                            transform=preprocess)
        imagenet_loader = torch.utils.data.DataLoader(imagenet_dataset, batch_size=args.batch_size, num_workers=2)
        zeroshot_weights, text_prompts = zeroshot_classifier(imagenet_classes, imagenet_templates, clip_model)
        acc = evaluate_zs(clip_model, zeroshot_weights, imagenet_loader)
        accuracy_per_model.append(acc)
        print(f"Model {model_name} Accuracy: {acc}")
        with open(f'{SAVE_DIR}/clip_model_results.txt', 'a') as f:
            f.write(f"Model {model_name} Accuracy: {acc}\n")

    average_accuracy = sum(accuracy_per_model) / len(model_names)
    print(f"Average Accuracy: {average_accuracy}")
    with open(f'{SAVE_DIR}/clip_model_results.txt', 'a') as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")