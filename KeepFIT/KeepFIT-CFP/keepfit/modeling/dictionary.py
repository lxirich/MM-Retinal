"""
数据的对照字典
definition:
This script contains dictionaries for expert knowledge prompts of several fundus image conditions for pre-training.

ensemble_prompts:
Also, it presents a dictionary for creating prompt ensembles for zero-shot classification and transferability.

datasets/abbreviations:
Finally, the script contains abbreviations of several relevant conditions used for FLAIR pre-training, and datasets names.
"""

# 类别名称对应的医学描述  每一个字典键值对：  类别名称str：医学描述str列表  The medical description corresponding to the category name
definitions = {"no diabetic retinopathy": ["no diabetic retinopathy", "no microaneurysms"],
                "mild diabetic retinopathy": ["only few microaneurysms"],
                "moderate diabetic retinopathy": ["many exudates near the macula",
                                                  "many haemorrhages near the macula",
                                                  "retinal thickening near the macula",
                                                  "hard exudates",
                                                  "cotton wool spots",
                                                  "few severe haemorrhages"],
                "severe diabetic retinopathy": ["venous beading",
                                                "many severe haemorrhages",
                                                "intraretinal microvascular abnormality"],
                "proliferative diabetic retinopathy": ["preretinal or vitreous haemorrhage",
                                                       "neovascularization"],
                "no referable diabetic macular edema": ["no apparent exudates"],
                "hard exudates": ["small white or yellowish deposits with sharp margins", "bright lesion"],
                "soft exudates": ["pale yellow or white areas with ill-defined edges", "cotton-wool spot",
                                  "small, whitish or grey, cloud-like, linear or serpentine, slightly elevated lesions"
                                  " with fimbriated edges"],
                "microaneurysms": ["small red dots"],
                "haemorrhages": ["dense, dark red, sharply outlined lesion"],
                "non clinically significant diabetic macular edema": ["presence of exudates outside the radius of one"
                                                                      " disc diameter from the macula center",
                                                                      "presence of exudates"],
                "age related macular degeneration": ["many small drusen", "few medium-sized drusen", "large drusen",
                                                     "macular degeneration"],
                "media haze": ["vitreous haze", "pathological opacity", "the obscuration of fundus details by vitreous"
                                                                        " cells and protein exudation"],
                "drusens": ["yellow deposits under the retina", "numerous uniform round yellow-white lesions"],
                "pathologic myopia": ["anomalous disc, macular atrophy and possible tessellation"],
                "branch retinal vein occlusion": ["occlusion of one of the four major branch retinal veins"],
                "tessellation": ["large choroidal vessels at the posterior fundus"],
                "epiretinal membrane": ["greyish semi-translucent avascular membrane"],
                "laser scar": ["round or oval, yellowish-white with variable black pigment centrally",
                               "50 to 200 micron diameter lesions"],
                "no laser scar": ["no laser scar"],
                "macular scar": ["macular scar"],
                "central serous retinopathy": ["subretinal fluid involving the fovea", "leakage"],
                "optic disc cupping": ["optic disc cupping"],
                "central retinal vein occlusion": ["central retinal vein occlusion"],
                "tortuous vessels": ["tortuous vessels"],
                "asteroid hyalosis": ["multiple sparking, yellow-white, and refractile opacities in the vitreous cavity",
                                      "vitreous opacities"],
                "optic disc pallor": ["pale yellow discoloration that can be segmental or generalized on optic disc"],
                "optic disc edema": ["optic disc edema"],
                "shunt": ["collateral vessels connecting the choroidal and the retinal vasculature",
                          "collateral vessels of large caliber and lack of leakage"],
                "anterior ischemic optic neuropathy": ["anterior ischemic optic neuropathy"],
                "parafoveal telangiectasia": ["parafoveal telangiectasia"],
                "retinal traction": ["retinal traction"],
                "retinitis": ["retinitis"],
                "chorioretinitis": ["chorioretinitis"],
                "exudates": ["small white or yellowish white deposits with sharp margins", "bright lesion"],
                "retinal pigment epithelium changes": ["retinal pigment epithelium changes"],
                "macular hole": ["lesion in the macula", "grayish fovea"],
                "retinitis pigmentosa": ["pigment deposits are present in the periphery"],
                "cotton wool spots": ["cotton wool spots", "soft exudates"],
                "colobomas": ["colobomas"],
                "optic disc pit maculopathy": ["optic disc pit maculopathy"],
                "preretinal haemorrhage": ["preretinal haemorrhage"],
                "myelinated nerve fibers": ["myelinated nerve fibers"],
                "haemorrhagic retinopathy": ["haemorrhagic retinopathy"],
                "central retinal artery occlusion": ["central retinal artery occlusion"],
                "tilted disc": ["tilted disc"],
                "cystoid macular edema": ["cysts in the macula region"],
                "post traumatic choroidal rupture": ["post traumatic choroidal rupture"],
                "choroidal folds": ["choroidal folds"],
                "vitreous haemorrhage": ["vitreous haemorrhage"],
                "macroaneurysm": ["macroaneurysm"],
                "vasculitis": ["vasculitis"],
                "branch retinal artery occlusion": ["branch retinal artery occlusion"],
                "plaque": ["plaque"],
                "haemorrhagic pigment epithelial detachment": ["haemorrhagic pigment epithelial detachment"],
                "collaterals": ["collaterals"],
                "normal": ["healthy", "no findings", "no lesion signs", "no glaucoma", "no retinopathy"],
                "large optic cup": ["abnormality in optic cup"],
                "retina detachment": ["retina detachment"],
                "Vogt-Koyanagi syndrome": ["Vogt-Koyanagi syndrome"],
                "maculopathy": ["maculopathy"],
                "glaucoma": ["optic nerve abnormalities", "abnormal size of the optic cup",
                             "anomalous size in the optic disc"],
                "optic atrophy": ["optic atrophy"],
                "severe hypertensive retinopathy": ["flame shaped hemorrhages at the disc margin, blurred disc margins,"
                                                    " congested retinal veins, papilledema, and secondary macular "
                                                    "exudates", "arterio-venous crossing changes, macular star and "
                                                                "cotton wool spots"],
                "disc swelling and elevation": ["disc swelling and elevation"],
                "dragged disk": ["dragged disk"],
                "congenital disk abnormality": ["disk abnormality", "optic disk lesion"],
                "Bietti crystalline dystrophy": ["Bietti crystalline dystrophy"],
                "peripheral retinal degeneration and break": ["peripheral retinal degeneration and break"],
                "neoplasm": ["neoplasm"],
                "yellow-white spots flecks": ["yellow-white spots flecks"],
                "fibrosis": ["fibrosis"],
                "silicon oil": ["silicon oil"],
                "no proliferative diabetic retinopathy": ["diabetic retinopathy with no neovascularization",
                                                          "no neovascularization"],
                "no glaucoma": ["no glaucoma"],
                "cataract": ["opacity in the macular area"],
                "hypertensive retinopathy": ["possible signs of haemorraghe with blot, dot, or flame-shaped",
                                             "possible presence of microaneurysm, cotton-wool spot, or hard exudate",
                                             "arteriolar narrowing", "vascular wall changes", "optic disk edema"],
                "neovascular age related macular degeneration": ["neovascular age-related macular degeneration"],
                "geographical age related macular degeneration": ["geographical age-related macular degeneration"],
                "acute central serous retinopathy": ["acute central serous retinopathy"],
                "chronic central serous retinopathy": ["chronic central serous retinopathy"],
                "no cataract": ["no cataract signs", "no obscure opacities"],
                "abnormal optic disc": ["abnormal optic disc"],
                "abnormal vessels": ["abnormal vessels"],
                "abnormal macula": ["abnormal macula"],
                "macular edema": ["macular edema"],
                "scar": ["scar"],
                "nevus": ["darkly pigmented lesion found in the back of the eye"],
                "increased cup disc": ["increased cup disc"],
                "intraretinal microvascular abnormalities": ["shunt vessels and appear as abnormal branching or"
                                                             " dilation of existing blood vessels (capillaries) "
                                                             "within the retina", "deeper in the retina than"
                                                             " neovascularization, has blurrier edges, is more"
                                                             " of a burgundy than a red, does not appear on the "
                                                             "optic disc", "vascular loops confined within the"
                                                             " retina"],
                "red small dots": ["microaneurysms"],
                "neovascularisation": ["neovascularisation"],
                "a disease": ["no healthy", "lesions"],
                "superficial haemorrhages": ["superficial haemorrhages"],
                "deep haemorrhages": ["deep haemorrhages"],
                "ungradable": ["no fundus", "very noisy", "noisy"],
                "noisy": ["noisy"],
                "normal macula": ["normal macula"],
                "macular degeneration": ["macular degeneration"],
                "diabetic retinopathy": ["diabetic retinopathy"],
                "no hypertensive retinopathy": ["no presence of hypertensive retinopathy"],
                "mild hypertensive retinopathy": ["mild arteriovenous ratio", "mild tortuosity",
                                                  "focal arteriolar narrowing",
                                                  "arteriovenous nicking"],
                "moderate hypertensive retinopathy": ["moderate arteriovenous ratio", "moderate tortuosity",
                                                      "cotton wool spots",
                                                      "flame-shaped haemorrhages"],
                "malignant hypertensive retinopathy": ["severe arteriovenous ratio", "severe tortuosity",
                                                       "swelling optical disk",
                                                       "flame-shaped haemorrhages"]
            }


# Datasets names
datasets = ["01_EYEPACS", "03_IDRID", "04_RFMid", "05_1000x39", "07_LAG", "09_PAPILA", "10_PARAGUAY", "12_ARIA",
            "14_AGAR300", "15_APTOS", "16_FUND-OCT", "17_DiaRetDB1", "18_DRIONS-DB", "19_Drishti-GS1", "20_E-ophta",
            "20_E-ophta", "21_G1020", "23_HRF", "24_ORIGA", "25_REFUGE", "26_ROC", "27_BRSET", "28_OIA-DDR",
            "02_MESIDOR", "05_20x3", "08_ODIR200x3", "13_FIVES"]


# 类别缩写  Category abbreviation
abbreviations = {"no diabetic retinopathy": "noDR", "mild diabetic retinopathy": "mildDR",
                 "moderate diabetic retinopathy": "modDR", "severe diabetic retinopathy": "sevDR",
                 "proliferative diabetic retinopathy": "prolDR", "diabetic macular edema": "DME",
                 "no referable diabetic macular edema": "noDME", "hard exudates": "hEX",
                 "soft exudates": "sEX", "microaneurysms": "MA", "haemorrhages": "HE",
                 "non clinically significant diabetic macular edema": "nonCSDME",
                 "age-related macular degeneration": "ARMD", "media haze": "MH", "drusens": "DN",
                 "pathologic myopia": "MYA", "branch retinal vein occlusion": "BRVO", "tessellation": "TSLN",
                 "epiretinal membrane": "ERM", "laser scar": "LS", "macular scar": "MS",
                 "central serous retinopathy": "CSR", "optic disc cupping": "ODC",
                 "central retinal vein occlusion": "CRVO", "tortuous vessels": "TV", "asteroid hyalosis": "AH",
                 "optic disc pallor": "ODP", "optic disc edema": "ODE",
                 "shunt": "ST", "anterior ischemic optic neuropathy": "AION", "parafoveal telangiectasia": "PT",
                 "retinal traction": "RT", "retinitis": "RS", "chorioretinitis": "CRS", "exudates": "EX",
                 "retinal pigment epithelium changes": "RPEC", "macular hole": "MHL", "retinitis pigmentosa": "RP",
                 "cotton wool spots": "CWS", "colobomas": "CB", "optic disc pit maculopathy": "ODM",
                 "preretinal haemorrhage": "PRH", "myelinated nerve fibers": "MNF", "haemorrhagic retinopathy": "HR",
                 "central retinal artery occlusion": "CRAO", "tilted disc": "TD", "cystoid macular edema": "CME",
                 "post traumatic choroidal rupture": "PTCR", "choroidal folds": "CF", "vitreous haemorrhage": "VH",
                 "macroaneurysm": "MCA", "vasculitis": "VS", "branch retinal artery occlusion": "BRAO", "plaque": "PLQ",
                 "haemorrhagic pigment epithelial detachment": "HPED", "collaterals": "CL", "normal": "N",
                 "large optic cup": "LOC", "retina detachment": "RD", "Vogt-Koyanagi syndrome": "VKH",
                 "maculopathy": "M", "glaucoma": "G", "optic atrophy": "OA", "severe hypertensive retinopathy": "sevHR",
                 "disc swelling and elevation": "DSE", "dragged disk": "DD", "congenital disk abnormality": "CDA",
                 "Bietti crystalline dystrophy": "BCD", "peripheral retinal degeneration and break": "PRDB",
                 "neoplasm": "NP", "yellow-white spots flecks": "YWSF", "fibrosis": "fibrosis", "silicon oil": "SO",
                 "no proliferative diabetic retinopathy": "noProlDR", "no glaucoma": "noG", "cataract": "CAT",
                 "hypertensive retinopathy": "HR", "neovascular age-related macular degeneration": "neovARMD",
                 "geographical age-related macular degeneration": "geoARMD",
                 "acute central serous retinopathy": "acCSR", "chronic central serous retinopathy": "chCSR",
                 "no cataract": "noCAT", "abnormal optic disc": "AOD", "abnormal vessels": "AV",
                 "abnormal macula": "AM", "macular edema": "ME", "scar": "S", "nevus": "NE",
                 "increased cup disc": "ICD", "intraretinal microvascular abnormalities": "IrMA",
                 "red small dots": "ReSD", "neovascularisation": "neoV", "a disease": "Dis"}