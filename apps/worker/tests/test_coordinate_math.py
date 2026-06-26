from app.utils import local_to_global_bbox


def test_standard_case():
    tile_bbox = (100.0, 200.0, 500.0, 600.0)
    local_bbox = (50.0, 60.0, 150.0, 180.0)

    global_bbox = local_to_global_bbox(local_bbox, tile_bbox)
    assert global_bbox == [150, 260, 250, 380]


def test_edge_case_top_left():
    tile_bbox = (150.0, 250.0, 450.0, 550.0)
    local_bbox = (0.0, 0.0, 80.0, 100.0)

    global_bbox = local_to_global_bbox(local_bbox, tile_bbox)
    assert global_bbox == [150, 250, 230, 350]


def test_edge_case_floats():
    tile_bbox = (100.2, 200.7, 500.9, 600.1)
    local_bbox = (50.5, 60.3, 150.8, 180.2)

    global_bbox = local_to_global_bbox(local_bbox, tile_bbox)
    # Rounding logic verification (Python banker's rounding):
    # tile_bbox -> tx_min = round(100.2) = 100, ty_min = round(200.7) = 201
    # local_bbox -> lx_min = round(50.5) = 50, ly_min = round(60.3) = 60
    # lx_max = round(150.8) = 151, ly_max = round(180.2) = 180
    # gx_min = 100 + 50 = 150
    # gy_min = 201 + 60 = 261
    # gx_max = 100 + 151 = 251
    # gy_max = 201 + 180 = 381
    assert global_bbox == [150, 261, 251, 381]
    assert all(isinstance(x, int) for x in global_bbox)
