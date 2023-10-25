import torch

class CycleColor:
    def __init__(self, epoch: int, cycle: int, background_tolerance: int, generation_range: int):
        self.epoch = epoch
        self.background_hue = ((self.epoch+10)*360/cycle)%360
        self.digit_hue = ((self.epoch+30)*360/cycle)%360
        self.generation_range = generation_range/cycle*360
        

        # self.background_pool = torch.as_tensor([colorsys.hsv_to_rgb(math.radians(i), 0.5, 0.5) for i in torch.linspace(self.background_hue, self.background_hue+generation_range, style_count)]).to(device)
        # self.digit_pool = torch.as_tensor([colorsys.hsv_to_rgb(math.radians(i), 0.5, 0.5) for i in torch.linspace(self.digit_hue+30, self.digit_hue+generation_range+30, style_count)]).to(device)
        # self.style_count = style_count
        self.background_tolerance = background_tolerance
        # self.rgb_background = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.background_hue), 0.5, 0.5)).to(device)
        # self.rgb_digit = torch.tensor(colorsys.hsv_to_rgb(math.radians(self.digit_hue), 0.5, 0.5)).to(device)
        # print(self.rgb_digit)
    def __call__(self, sample: torch.tensor):
        single = False

        if(sample.dim() ==3):
            sample = sample.unsqueeze(0)
            single = True

        n = sample.shape[0]

        max = torch.max(sample)
        min = torch.min(sample)
        image = (sample-min)/(max-min)

        bg_saturation, bg_val, digit_saturation, digit_val = torch.FloatTensor(4, n).uniform_(0.2, 0.9)
        bg_hue = torch.FloatTensor(n).uniform_(self.background_hue,self.background_hue+self.generation_range)
        # bg_hue = torch.linspace(self.background_hue, self.background_hue+self.generation_range, n)
        digit_hue = torch.FloatTensor(n).uniform_(self.background_hue+30, self.background_hue+self.generation_range+30)

        # digit_hue = torch.linspace(self.digit_hue+30, self.digit_hue+self.generation_range+30, n)

        bg_rgb = hsv_to_rgb((torch.sin(torch.deg2rad(bg_hue))+1)/2, bg_saturation, bg_val).to(sample.device)
        digit_rgb = hsv_to_rgb((torch.sin(torch.deg2rad(digit_hue))+1)/2, digit_saturation, digit_val).to(sample.device)
        

        background_mask = (image < image.min()+self.background_tolerance) & (image > image.min()-self.background_tolerance)
        digit_mask = ~background_mask
        color_pic = image.repeat_interleave(3, dim=1)
        permuted_color_pic = color_pic.permute((0,2,3,1))
        
        bg_interleave_factors = background_mask.squeeze(dim=1).sum(dim=(1,2))
        digit_interleave_factors = digit_mask.squeeze(dim=1).sum(dim=(1,2))

        # color_indices = torch.randint(0,self.style_count, (sample.shape[0],))
        # print(self.background_pool[color_indices].shape, bg_interleave_factors.shape)
        # bg_colors = self.background_pool[color_indices].repeat_interleave(bg_interleave_factors, dim=0)
        # digit_colors = self.digit_pool[color_indices].repeat_interleave(digit_interleave_factors, dim=0)
        bg_colors = bg_rgb.repeat_interleave(bg_interleave_factors, dim=0)
        digit_colors =  digit_rgb.repeat_interleave(digit_interleave_factors, dim=0)

        # print(bg_colors.shape, permuted_color_pic[background_mask.squeeze()].shape)

        permuted_color_pic[background_mask.squeeze(dim=1)]  = bg_colors
        permuted_color_pic[digit_mask.squeeze(dim=1)]  = digit_colors
        
        if single:
            color_pic = color_pic.squeeze(dim=0)

        return color_pic, torch.deg2rad(bg_hue)


def hsv_to_rgb(h, s, v):
    i = (h*6.0).int() # XXX assume int() truncates!
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6

    res = torch.zeros((len(h),3))

    ind = i==0
    res[ind] = torch.stack((v[ind], t[ind], p[ind])).T
    ind = i==1
    res[ind] = torch.stack((q[ind], v[ind], p[ind])).T
    ind = i==2
    res[ind] = torch.stack((p[ind], v[ind], t[ind])).T
    ind = i==3
    res[ind] = torch.stack((p[ind], q[ind], v[ind])).T
    ind = i==4
    res[ind] = torch.stack((t[ind], p[ind], v[ind])).T
    ind = i==5
    res[ind] = torch.stack((v[ind], p[ind], q[ind])).T

    return res