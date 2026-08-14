# -*- coding: utf-8 -*-
"""
MONO down-looking Mako G-234 mount ten an item Profil 5 60x20 - every part is
exported INDIVIDUALLY (.stp + .stl) because all of them get 3D-printed,
including a printable stand-in for the item profile.

Parts (each its own file):
  1. part_cam_holder   - foot plate (lens aperture) + vertical back plate
                         (camera screwed ten with 2x M3) + top flange (2x M5
                         into the profile's bottom T-slots).  40 mm cable
                         headroom above the camera back for the RJ45 plug.
  2. part_cam_box      - open protective U-box: open top (Ethernet cable),
                         open back (slides over the camera against the plate),
                         fixed with 3x M3 from below through the foot.
  3. part_item_profil_60x20 - printable Line-5 profile piece (default 200 mm).
  4. mono_cam_assembly - everything together, for reference only.

Camera dims from the Mako G drawing (user-supplied datasheet layout):
  body 29.2 x 29.2 x 54.7 (60.5 incl. rear connectors), front snout D27.
  Camera mounting uses the THREE M3 (depth 2.3) ten the top face:
    1 hole ten the centerline at 23.7 from the front face,
    2 holes 20 mm apart (x=+-10) at 23.7+15=38.7 from the front face.
  (Reading of the 23.7|15 chain: if the drawing means 15-from-REAR instead,
  all three shift 7.7 mm -> change M3_SINGLE/M3_PAIR only.)
  -> use M3x6 screws (4 mm plate + ~2 mm engagement; threads only 2.3 deep,
  longer screws bottom out).  The 2x M3 ten the bottom face (12.2/42.6) stay
  free as a spare pattern.

Coordinate system: camera axis vertical, lens DOWN.  z=0 is the camera front
face resting ten the holder foot; the profile runs along X above everything.
Units mm.  B-rep via gmsh OpenCASCADE.
"""
import gmsh 

DIR ="C:/Users/DE00024082/Desktop/code/"

# ---------------- camera (Mako G-234, from the drawing) --------------------
CAM_W =29.2 # square body cross-section
CAM_BL =54.7 # body length (front face -> back plate)
CAM_TL =60.5 # incl. rear connector block
SNOUT_D ,SNOUT_L =27.0 ,5.8 
M3_SINGLE =23.7 # top-face single M3: centerline, above the front face
M3_PAIR =38.7 # top-face M3 pair: 23.7+15, at x=+-PAIR_DX
PAIR_DX =10.0 # pair half-spacing (20 mm apart, drawing dim "20")
M3_DEPTH =2.3 # thread depth in the camera -> M3x6 screws!

# ---------------- holder ---------------------------------------------------
FIT =0.2 # camera-to-plate fit clearance
PT =4.0 # plate/foot thickness
FOOT =44.0 # foot plate is FOOT x FOOT
AP_D =28.0 # lens aperture (snout D27 -> 0.5 radial)
HEAD =40.0 # cable headroom above camera back (RJ45 + bend)
FLT =5.0 # top flange thickness
FLY =40.0 # flange depth (Y) - SINGLE-slot mounting: both M5
FLX =14.0 #   sit IN-LINE in one groove, at x=+-FLX (anti-twist
#   comes from two screws along the same slot)
M3C ,M5C =3.4 ,5.5 # clearance holes
YIN =CAM_W /2 +FIT # plate inner face y = 14.8
ZFL =CAM_TL +HEAD # flange bottom = 100.5

# ---------------- box (open protective cover) ------------------------------
BW =4.0 # wall thickness
GAP =1.0 # camera-to-wall clearance
BH =62.0 # wall height (covers the 60.5 body)
XIN =CAM_W /2 +GAP # inner half-width = 15.6
PILOT =2.7 # M3 self-tap pilot in the wall bottoms

# ---------------- item Profil 5 60x20 (printable) --------------------------
PROF_L =200.0 # printable length (real profile: buy & cut any length)
PW ,PH =60.0 ,20.0 
SLOT_W ,SLOT_WC ,SLOT_TH ,SLOT_D =5.0 ,6.3 ,1.8 ,6.35 
CORE_D =4.3 
PZ0 =ZFL +FLT # profile bottom sits ten the flange top = 105.5

# ---------------- NEMA 17 motor, HORIZONTAL axis + laser scanner -----------
# The motor shaft is horizontal (90 deg to the camera axis); the line-laser
# module is clamped 90 deg to the shaft.  Rotating the shaft tilts the fan ->
# the laser LINE sweeps across the part under the camera.
N17_W =42.3 # square faceplate
N17_L =40.0 # body length (typ. 17HS4x)
N17_HOLE =31.0 # M3 mounting square -> holes at +-15.5
N17_BOSS =22.0 # centering boss dia (2 mm high)
N17_SH_D ,N17_SH_L =5.0 ,24.0 
MSH_Z =45.0 # shaft axis height
MPLT =5.0 # motor plate thickness (vertical, in the XZ plane)
MPY0 =10.0 # plate front face y (plate y = 10..15, motor behind)
MPW =52.0 # plate width
MOTOR_X =65.0 # motor position along the profile (camera at x=0)
LAS_D =12.2 # laser module clamp bore (D12 module + 0.2)

# ---------------- fasteners (modelled, assembly only) -----------------------
M5_HD ,M5_HH ,M5_D =8.5 ,5.0 ,5.0 
M3_HD ,M3_HH ,M3_D =5.5 ,3.0 ,3.0 

occ =None 


# ============================ part builders ================================
def make_screw (cx ,cy ,cz ,dx ,dy ,dz ,hd ,hh ,sd ,sl ):
    head =occ .addCylinder (cx ,cy ,cz ,dx *hh ,dy *hh ,dz *hh ,hd /2 )
    shaft =occ .addCylinder (cx ,cy ,cz ,dx *(hh +sl ),dy *(hh +sl ),dz *(hh +sl ),sd /2 )
    r ,_ =occ .fuse ([(3 ,head )],[(3 ,shaft )])
    return r [0 ][1 ]


def make_tnut (cx ,cy ,cz ):
    b =occ .addBox (cx -5 ,cy -3 ,cz -1.75 ,10 ,6 ,3.5 )
    h =occ .addCylinder (cx ,cy ,cz -2.0 ,0 ,0 ,4.0 ,M5_D /2 )
    r ,_ =occ .cut ([(3 ,b )],[(3 ,h )])
    return r [0 ][1 ]


def build_holder ():
    parts =[
    (3 ,occ .addBox (-FOOT /2 ,-FOOT /2 ,-PT ,FOOT ,FOOT ,PT )),# foot
    (3 ,occ .addBox (-FOOT /2 ,YIN ,-PT ,FOOT ,PT ,ZFL +PT )),# back plate
    (3 ,occ .addBox (-FOOT /2 ,-FLY /2 ,ZFL ,FOOT ,FLY ,FLT )),# top flange
    (3 ,occ .addBox (6.0 ,YIN +PT ,-PT ,4.0 ,5.0 ,ZFL +PT +FLT )),# rib R
    (3 ,occ .addBox (-10.0 ,YIN +PT ,-PT ,4.0 ,5.0 ,ZFL +PT +FLT )),# rib L
    ]
    r ,_ =occ .fuse ([parts [0 ]],parts [1 :])
    tag =r [0 ][1 ]
    cut =[(3 ,occ .addCylinder (0 ,0 ,-PT -1 ,0 ,0 ,PT +2 ,AP_D /2 ))]# aperture
    for hx ,hz in ((0.0 ,M3_SINGLE ),(PAIR_DX ,M3_PAIR ),(-PAIR_DX ,M3_PAIR )):
        cut .append ((3 ,occ .addCylinder (hx ,YIN -1 ,hz ,0 ,PT +2 ,0 ,M3C /2 )))# 3x camera M3
    for sx in (-FLX ,FLX ):# flange M5, ONE slot, in-line
        cut .append ((3 ,occ .addCylinder (sx ,0 ,ZFL -1 ,0 ,0 ,FLT +2 ,M5C /2 )))
    for hx ,hy in ((0 ,-XIN -BW /2 ),(XIN +BW /2 ,-2.4 ),(-XIN -BW /2 ,-2.4 )):# box M3
        cut .append ((3 ,occ .addCylinder (hx ,hy ,-PT -1 ,0 ,0 ,PT +2 ,M3C /2 )))
    r ,_ =occ .cut ([(3 ,tag )],cut )
    return r [0 ][1 ]


def build_box ():
    parts =[
    (3 ,occ .addBox (-XIN -BW ,-XIN -BW ,0 ,2 *(XIN +BW ),BW ,BH )),# front wall
    (3 ,occ .addBox (XIN ,-XIN -BW ,0 ,BW ,XIN +BW +YIN ,BH )),# side R
    (3 ,occ .addBox (-XIN -BW ,-XIN -BW ,0 ,BW ,XIN +BW +YIN ,BH )),# side L
    ]
    r ,_ =occ .fuse ([parts [0 ]],parts [1 :])
    tag =r [0 ][1 ]
    cut =[]
    for hx ,hy in ((0 ,-XIN -BW /2 ),(XIN +BW /2 ,-2.4 ),(-XIN -BW /2 ,-2.4 )):
        cut .append ((3 ,occ .addCylinder (hx ,hy ,-1 ,0 ,0 ,11.0 ,PILOT /2 )))
    r ,_ =occ .cut ([(3 ,tag )],cut )
    return r [0 ][1 ]


def build_profile ():
    prof =occ .addBox (-PROF_L /2 ,-PW /2 ,PZ0 ,PROF_L ,PW ,PH )
    xln =PROF_L +2.0 
    pc =[]

    def slot_z (gy ,surf_z ,into ):
        tz =surf_z -SLOT_TH if into <0 else surf_z 
        cz =surf_z -SLOT_D if into <0 else surf_z +SLOT_TH 
        return [(3 ,occ .addBox (-PROF_L /2 -1 ,gy -SLOT_W /2 ,tz ,xln ,SLOT_W ,SLOT_TH )),
        (3 ,occ .addBox (-PROF_L /2 -1 ,gy -SLOT_WC /2 ,cz ,xln ,SLOT_WC ,SLOT_D -SLOT_TH ))]

    def slot_y (surf_y ,into ,zc ):
        ty =surf_y if into >0 else surf_y -SLOT_TH 
        cy =surf_y +SLOT_TH if into >0 else surf_y -SLOT_D 
        return [(3 ,occ .addBox (-PROF_L /2 -1 ,ty ,zc -SLOT_W /2 ,xln ,SLOT_TH ,SLOT_W )),
        (3 ,occ .addBox (-PROF_L /2 -1 ,cy ,zc -SLOT_WC /2 ,xln ,SLOT_D -SLOT_TH ,SLOT_WC ))]

    for gy in (-20.0 ,0.0 ,20.0 ):
        pc +=slot_z (gy ,PZ0 +PH ,-1 )# top face
        pc +=slot_z (gy ,PZ0 ,+1 )# bottom face
    pc +=slot_y (-PW /2 ,+1 ,PZ0 +PH /2 )# front face
    pc +=slot_y (+PW /2 ,-1 ,PZ0 +PH /2 )# back face
    for gy in (-20.0 ,0.0 ,20.0 ):
        pc .append ((3 ,occ .addCylinder (-PROF_L /2 ,gy ,PZ0 +PH /2 ,PROF_L ,0 ,0 ,CORE_D /2 )))
    r ,_ =occ .cut ([(3 ,prof )],pc )
    return r [0 ][1 ]


def build_motor_holder (ox =0.0 ):
    """Vertical plate hanging from the flange; the NEMA 17 face bolts against
    its BACK (4x M3 from the front), shaft horizontal through the D23 hole.
    Motor body behind the plate, laser rotor in front.  Single-slot flange
    like the camera holder.  Print flat ten the plate, no supports."""
    parts =[
    (3 ,occ .addBox (ox -MPW /2 ,-FLY /2 ,ZFL ,MPW ,FLY ,FLT )),# flange
    (3 ,occ .addBox (ox -MPW /2 ,MPY0 ,20.0 ,MPW ,MPLT ,ZFL -20.0 )),# plate
    (3 ,occ .addBox (ox +MPW /2 -4.0 ,MPY0 -5.0 ,20.0 ,4.0 ,5.0 ,ZFL -20.0 )),# rib R
    (3 ,occ .addBox (ox -MPW /2 ,MPY0 -5.0 ,20.0 ,4.0 ,5.0 ,ZFL -20.0 )),# rib L
    ]
    r ,_ =occ .fuse ([parts [0 ]],parts [1 :])
    tag =r [0 ][1 ]
    cut =[(3 ,occ .addCylinder (ox ,MPY0 -1 ,MSH_Z ,0 ,MPLT +2 ,0 ,N17_BOSS /2 +0.5 ))]
    for hx in (-N17_HOLE /2 ,N17_HOLE /2 ):# 4x M3 (motor face pattern)
        for hz in (-N17_HOLE /2 ,N17_HOLE /2 ):
            cut .append ((3 ,occ .addCylinder (ox +hx ,MPY0 -1 ,MSH_Z +hz ,0 ,MPLT +2 ,0 ,M3C /2 )))
    for sx in (-FLX ,FLX ):# flange M5, ONE slot, in-line
        cut .append ((3 ,occ .addCylinder (ox +sx ,0 ,ZFL -1 ,0 ,0 ,FLT +2 ,M5C /2 )))
    r ,_ =occ .cut ([(3 ,tag )],cut )
    return r [0 ][1 ]


def build_laser_holder (ox =0.0 ,oz =0.0 ):
    """PRINTED part: clamp hub ten the motor shaft (D5.2 bore + M3 set screw)
    + radial arm + clamp ring whose bore is 90 deg to the shaft axis for a
    D12 line-laser module (slit + M3 pinch screw).  Local origin = shaft axis."""
    hub =occ .addCylinder (ox ,-8.0 ,oz ,0 ,8.0 ,0 ,7.0 )
    arm =occ .addBox (ox -5.0 ,-8.0 ,oz -30.0 ,10.0 ,8.0 ,30.0 )
    ring =occ .addCylinder (ox ,-4.0 ,oz -48.0 ,0 ,0 ,18.0 ,9.0 )
    r ,_ =occ .fuse ([(3 ,hub )],[(3 ,arm ),(3 ,ring )])
    tag =r [0 ][1 ]
    cut =[
    (3 ,occ .addCylinder (ox ,-9.0 ,oz ,0 ,10.0 ,0 ,2.6 )),# shaft bore D5.2
    (3 ,occ .addCylinder (ox ,-4.0 ,oz +8.0 ,0 ,0 ,-6.0 ,1.25 )),# set-screw pilot
    (3 ,occ .addCylinder (ox ,-4.0 ,oz -49.0 ,0 ,0 ,20.0 ,LAS_D /2 )),# module bore
    (3 ,occ .addBox (ox -1.5 ,-13.5 ,oz -48.5 ,3.0 ,9.5 ,19.0 )),# clamp slit
    (3 ,occ .addCylinder (ox -8.0 ,-9.5 ,oz -39.0 ,16.0 ,0 ,0 ,1.25 )),# pinch pilot
    ]
    r ,_ =occ .cut ([(3 ,tag )],cut )
    return r [0 ][1 ]


def build_motor (ox =MOTOR_X ):
    """NEMA 17 with HORIZONTAL shaft through the plate (reference only)."""
    yf =MPY0 +MPLT # motor face = plate back
    body =occ .addBox (ox -N17_W /2 ,yf ,MSH_Z -N17_W /2 ,N17_W ,N17_L ,N17_W )
    boss =occ .addCylinder (ox ,yf ,MSH_Z ,0 ,-2.0 ,0 ,N17_BOSS /2 )
    shaft =occ .addCylinder (ox ,yf ,MSH_Z ,0 ,-N17_SH_L ,0 ,N17_SH_D /2 )
    r ,_ =occ .fuse ([(3 ,body )],[(3 ,boss ),(3 ,shaft )])
    return r [0 ][1 ]


def build_laser_module (ox =MOTOR_X ):
    """D12 line-laser module sitting in the clamp, pointing DOWN at rest."""
    return occ .addCylinder (ox ,-4.0 ,MSH_Z -58.0 ,0 ,0 ,30.0 ,6.0 )


def build_motor_fasteners (ox =MOTOR_X ):
    for sx in (-FLX ,FLX ):# holder -> profile (M5, 1 slot)
        make_tnut (ox +sx ,0 ,PZ0 +SLOT_TH +1.75 )
        make_screw (ox +sx ,0 ,ZFL -M5_HH ,0 ,0 ,1 ,M5_HD ,M5_HH ,M5_D ,10.5 )
    for hx in (-N17_HOLE /2 ,N17_HOLE /2 ):# motor -> plate (4x M3x8)
        for hz in (-N17_HOLE /2 ,N17_HOLE /2 ):
            make_screw (ox +hx ,MPY0 -M3_HH ,MSH_Z +hz ,0 ,1 ,0 ,
            M3_HD ,M3_HH ,M3_D ,MPLT +4.0 )


def build_camera ():
    body =occ .addBox (-CAM_W /2 ,-CAM_W /2 ,0 ,CAM_W ,CAM_W ,CAM_BL )
    rear =occ .addBox (-11.35 ,-11.35 ,CAM_BL ,22.7 ,22.7 ,CAM_TL -CAM_BL )
    snout =occ .addCylinder (0 ,0 ,0 ,0 ,0 ,-SNOUT_L ,SNOUT_D /2 )
    r ,_ =occ .fuse ([(3 ,body )],[(3 ,rear ),(3 ,snout )])
    return r [0 ][1 ]


def build_fasteners ():
    for sx in (-FLX ,FLX ):# holder -> profile (M5, 1 slot)
        make_tnut (sx ,0 ,PZ0 +SLOT_TH +1.75 )
        make_screw (sx ,0 ,ZFL -M5_HH ,0 ,0 ,1 ,M5_HD ,M5_HH ,M5_D ,10.5 )
    for hx ,hz in ((0.0 ,M3_SINGLE ),(PAIR_DX ,M3_PAIR ),(-PAIR_DX ,M3_PAIR )):
        make_screw (hx ,YIN +PT +M3_HH ,hz ,0 ,-1 ,0 ,# camera -> holder (3x M3x6)
        M3_HD ,M3_HH ,M3_D ,PT +M3_DEPTH )
    for hx ,hy in ((0 ,-XIN -BW /2 ),(XIN +BW /2 ,-2.4 ),(-XIN -BW /2 ,-2.4 )):# box (M3)
        make_screw (hx ,hy ,-PT -M3_HH ,0 ,0 ,1 ,M3_HD ,M3_HH ,M3_D ,PT +8.0 )


        # ============================ export driver ================================
def export (stem ,builders ,stl_size =2.0 ):
    global occ 
    gmsh .initialize ()
    gmsh .option .setNumber ("General.Terminal",0 )
    try :
        gmsh .option .setString ("Geometry.OCCTargetUnit","MM")
    except Exception :
        pass 
    gmsh .model .add (stem )
    occ =gmsh .model .occ 
    for b in builders :
        b ()
    occ .synchronize ()
    n =len (gmsh .model .getEntities (3 ))
    g =gmsh .model .getBoundingBox (-1 ,-1 )
    gmsh .write (DIR +stem +".stp")
    gmsh .option .setNumber ("Mesh.MeshSizeMax",stl_size )
    gmsh .model .mesh .generate (2 )
    gmsh .write (DIR +stem +".stl")
    gmsh .finalize ()
    print ("%-28s %2d solid(s)  %6.1f x %5.1f x %5.1f mm"
    %(stem ,n ,g [3 ]-g [0 ],g [4 ]-g [1 ],g [5 ]-g [2 ]))


if __name__ =="__main__":
    print ("== individual printable parts ==")
    export ("part_cam_holder",[build_holder ])
    export ("part_cam_box",[build_box ])
    export ("part_motor_holder",[build_motor_holder ])
    export ("part_laser_holder",[build_laser_holder ])
    export ("part_item_profil_60x20",[build_profile ])
    print ("== reference assembly ==")
    export ("mono_cam_assembly",
    [build_profile ,build_holder ,build_box ,build_camera ,build_fasteners ,
    lambda :build_motor_holder (MOTOR_X ),build_motor ,
    lambda :build_laser_holder (MOTOR_X ,MSH_Z ),build_laser_module ,
    build_motor_fasteners ],
    stl_size =3.0 )
    print ("done.")
