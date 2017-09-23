-- demo
require 'nn'
require 'cudnn'
require 'image'

-- Load model

model  =torch.load('food-net.t7')
-- Load Category names
categories  =torch.load('categories.t7')




cmd = torch.CmdLine()
 
cmd:text('Classify a food image to one of the 101 classes of food-101')
cmd:text()
cmd:text('Options')
cmd:option('-imfile','apple_pie.jpg','input image')
cmd:text()

-- parse input params
params = cmd:parse(arg)




model:evaluate()

-- load sample test image
im = image.load(params.imfile,3,'float');
if im:size(3) < im:size(2) then
im = image.scale(im, 256, 256 * im:size(2) / im:size(3))
else
im = image.scale(im, 256 * im:size(3) / im:size(2), 256 )
end

-- preprocess image (scale and crop)
 
--  random crop(s)- allow this option later
if(1) then
h1 = math.ceil(torch.uniform(1e-2, im:size(2)-224))
w1 = math.ceil(torch.uniform(1e-2, im:size(3)-224))
im = image.crop(im, w1, h1, w1 + 224, h1 + 224)
image.display(im)
 else
	 
	h1 =  (im:size(2)-224)/2
	w1 =  (im:size(3)-224)/2	
	im = image.crop(im, w1, h1, w1 + 224, h1 + 224)
	image.display(im)

 end
  
--remove mean and divide with std 
 mean={0.55511024474619, 0.44239095055101,0.33011610164876}
stdval={ 0.22127111256688,0.23319352981181, 0.23024111427517}
 for i=1,3 do 
   im[i]:add(-mean[i]) 
   im[i]:div(stdval[i])
end

t= torch.CudaTensor(1,3,224,224)
-- copy image to 1-batch tensor
t[1] = im:clone()
-- get net response
prob  = model(t)

-- convert to probabilities!
prob =  prob:exp()
--  find max
val, id = prob:sort(1,true)


print("top matched categories:")
for i=1,5 do
print("",categories[id[i]]," probability",val[i])
end


