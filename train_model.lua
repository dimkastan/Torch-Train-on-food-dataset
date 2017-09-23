--[[

Torch implementation of squeezenet for food recognition.

Author: Dimitris Kastaniotis

Contact me for any comments, suggestions etc 
 

--]]

require 'image'
require 'nn'
require 'optim'
require 'xlua'
require 'cunn'
require 'cudnn'
require 'torch'
require 'cutorch'


torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(2)


-- Options
opt={cuda=true}
--
opt.dir="food-101/images"; --path to images
opt.ext="jpg"


-- define the output of squeezenet
nClasses=101;
WIDTH = 224
HEIGHT = 224;
-- max number of images per category
NumImages= 1000;
epoch =0;
batchSize = 20
Epochs=250;
LR = 0.01;
wInit  ='xavier'

-- sgd params
  sgdState = sgdState or {

   learningRate =  LR,--opt.learningRate,
   momentum = 0.9, --opt.momentum,
   weightDecay =5e-4,
   dampening = 0.0,
   learningRateDecay = 0.0

  }


-- network type: (This is taken from https://github.com/soumith/imagenet-multiGPU.torch You can use any model from there,
-- but consider to use an approprirate weightDecay
-- initialization scheme)
local function fire(ch, s1, e1, e3)
        local net = nn.Sequential()
        net:add(nn.SpatialConvolution(ch, s1, 1, 1))
        net:add(nn.ReLU(true))
        local exp = nn.Concat(2)
        exp:add(nn.SpatialConvolution(s1, e1, 1, 1))
        exp:add(nn.SpatialConvolution(s1, e3, 3, 3, 1, 1, 1, 1))
        net:add(exp)
        net:add(nn.ReLU())
        return net
end


local function bypass(net)
        local cat = nn.ConcatTable()
        cat:add(net)
        cat:add(nn.Identity())
        local seq = nn.Sequential()
        seq:add(cat)
        seq:add(nn.CAddTable(true))
        return seq
end

local function squeezenet(output_classes, w, h)
        local net = nn.Sequential()

        net:add(nn.SpatialConvolution(3, 96, 7, 7, 2, 2, 0, 0)) -- conv1
        net:add(nn.ReLU(true))
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(fire(96, 16, 64, 64))  --fire2
        net:add(bypass(fire(128, 16, 64, 64)))  --fire3
        net:add(fire(128, 32, 128, 128))  --fire4
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(bypass(fire(256, 32, 128, 128)))  --fire5
        net:add(fire(256, 48, 192, 192))  --fire6
        net:add(bypass(fire(384, 48, 192, 192)))  --fire7
        net:add(fire(384, 64, 256, 256))  --fire8
        net:add(nn.SpatialMaxPooling(3, 3, 2, 2))
        net:add(bypass(fire(512, 64, 256, 256)))  --fire9
        net:add(nn.Dropout(0.5))
        net:add(nn.SpatialConvolution(512, output_classes, 1, 1, 1, 1, 1, 1)) --conv10
        net:add(nn.ReLU())
        -- add this for automatic determination
        --out = net:forward(torch.randn(1,3,h,w));
       -- net:add(nn.SpatialAveragePooling(out:size(3),out:size(4), 1, 1))

        net:add(nn.SpatialAveragePooling(14, 14, 1, 1))
        net:add(nn.View(output_classes))
        net:add(nn.LogSoftMax())
        -- net:add(nn.View(output_classes))
        --net:add(nn.Linear(output_classes,output_classes))


        return net
end

function createModel(nClasses,w,h) --(nGPU)
    local model = squeezenet(nClasses,w,h)

    return model
end






-- Create SqueezeNet

model = createModel(nClasses,WIDTH, HEIGHT)


-- kaiming normalization
if (wInit == 'kaiming') then
for indx,module in pairs(model:findModules('nn.SpatialConvolution')) do
   module.weight:normal(0,math.sqrt(2/(module.kW*module.kH*module.nOutputPlane)))
end
end
if (wInit =='xavier') then
  --xavier initialization
  for indx,module in pairs(model:findModules('nn.SpatialConvolution')) do
     module.weight:normal(0,math.sqrt(1/(module.kW*module.kH*module.nOutputPlane)))
  end

end


-- train and test confusion matrices
confusion = optim.ConfusionMatrix(nClasses)
confusiontest = optim.ConfusionMatrix(nClasses)
-- use subrange as table.move is not available in some LUA versions

catnames={}
catids={}
cats=0;
 function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

-- list for all files
allfiles={}
for dirs in paths.iterdirs(opt.dir) do
    if(cats<nClasses) then
print('User',dirs)
  ta = paths.dir(table.concat({opt.dir, "/",dirs}));
  if(#ta>2) then -- if directory has images
    cats=cats+1;
    table.insert(catnames, dirs)
    table.insert(catids, cats)
    --table.insert(allfiles, table.move(ta,3,2+NumImages,1,{}))
    table.insert(allfiles, subrange(ta,3,2+NumImages))
  end
  end
end


-- create train logger
trainLogger = optim.Logger(paths.concat('./','trainloss_sqreslog'))
testLogger  = optim.Logger(paths.concat('./','testloss_sqres.log'))

trainLoggerAcc = optim.Logger(paths.concat('./', 'train_sqres.log'))
testLoggerAcc = optim.Logger(paths.concat('./', 'test_sqres.log'))



criterion = nn.ClassNLLCriterion()




if(opt.cuda) then
--model:cuda()
model:cuda()
cudnn.convert(model,cudnn)
cudnn.fastest = true

end

if(opt.cuda) then
criterion:cuda() -- Later move to cuda using an input param
end




print(model)




temp_files={}
temp_labels={}
for i=1,#allfiles do
for j=1,#allfiles[i] do
    print(table.concat({catnames[i],"/",allfiles[i][j]}))
table.insert(temp_files, table.concat({catnames[i],"/",allfiles[i][j]}))
table.insert(temp_labels,{i})
end
end

--shuffle first (in order to sample all classes)
sf=torch.randperm(#temp_files)

files={}
labels ={}
for i=1,#temp_files do
  table.insert(files,temp_files[sf[i]]);
  table.insert(labels,temp_labels[sf[i]]);
end

--clear
temp_files = nil
temp_labels = nil



-- split rain and test
Ntrain = torch.ceil(#files*0.9)
Ntest =  #files - Ntrain;
train={}
test={}
train=subrange(files,1,Ntrain)
trlbls=subrange(labels,1,Ntrain)
test=subrange(files,Ntrain+1,Ntrain+Ntest)
tslbls=subrange(labels,Ntrain+1,Ntrain+Ntest)



-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()




local function train_epoch()


  model:training()
  local aveloss=0.0
  local counter =0;

  -- shuffle in every epoch
  trainLBLS=torch.Tensor(batchSize)
  inputs = torch.CudaTensor(batchSize,3,HEIGHT, WIDTH)

  strain=torch.randperm(#train)
   -- Iterate over all images
  K = torch.floor(#train);

  km=#train % batchSize; -- process the rest files.. (add this later)
   for p=1, K-batchSize,batchSize do


   -- load one image per person
      for jj=1,batchSize do
        local fn = table.concat({opt.dir,"/",train[strain[p+jj]]});
        trainLBLS[jj]=trlbls[strain[p+jj]][1];
         -- print(fn)
          -- make smallest side equal to 256
        im = image.load(fn,3,'float')

        if im:size(3) < im:size(2) then
          im = image.scale(im, 256, 256 * im:size(2) / im:size(3))
        else
          im = image.scale(im, 256 * im:size(3) / im:size(2), 256 )
        end

        local input = im


         local iW = input:size(3)
         local iH = input:size(2)

         -- do random crop
         local oW = WIDTH
         local oH = HEIGHT
         local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
         local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
         local tim = image.crop(input, w1, h1, w1 + oW, h1 + oH)
         assert(tim:size(3) == oW)
         assert(tim:size(2) == oH)


          -- remove mean (estimated on a subset of imagenet)
          local mean={0.55511024474619, 0.44239095055101,0.33011610164876}
          local stdval={ 0.22127111256688,0.23319352981181, 0.23024111427517}
          for i=1,3 do
            tim[i]:add(-mean[i])
            tim[i]:div(stdval[i])
          end



       if(torch.uniform()>0.5) then
       tim=image.hflip(tim)
       end



    inputs[jj]=tim:clone()


    end




  local f, outputs;
  -- create closure to evaluate f(X) and df/dX
  feval = function(x)
  local  lbls=trainLBLS:cuda()

     -- just in case:
       collectgarbage()

      -- reset gradients
       gradParameters:zero()

       outputs = model:forward(inputs)

       f = criterion:forward(outputs, lbls )
         -- estimate df/dW
     local df_do = criterion:backward(outputs, lbls)
     model:backward(inputs , df_do)

    -- update confusion
         for i = 1,batchSize do
            confusion:add(outputs[i], lbls[i])

         end

         -- return f and df/dX
         return f,gradParameters

  end


  _,fs= optim.sgd(feval, parameters, sgdState)
  --print(sgdState)

  -- disp progress
  xlua.progress(p,K)


  aveloss=aveloss+fs[1]
  counter=counter+1;

 end
  -- first print confusion and then add validation values
 print(confusion)



 trainLogger:add{ ['Loss ='] = aveloss/counter }
 print("Average loss = ",aveloss/counter )
 trainLogger:style{['Loss ='] = '-'}
 trainLogger:plot()




 trainLoggerAcc:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}


 trainLoggerAcc:style{['% mean class accuracy (train set)'] = '-'}
 trainLoggerAcc:plot()
 confusion:zero() -- reset per epoch

  --sgdState.learningRate=sgdState.learningRate*0.995
  sgdState.learningRate =LR*0.1^math.floor(epoch/10)
  -- evaluate test set...


  --model:clearState()
  snapshot={model,epoch,sgdState}
  fname  =table.concat({"./","MC_modelState.t7"})
  print('saving model',model)
  torch.save(fname,model)
end



local function test_epoch()

  model:evaluate()

  local aveloss=0.0
  local counter =0;

    -- shuffle in every epoch
  testLBLS=torch.Tensor(batchSize)
  local inputs = torch.CudaTensor(batchSize,3,HEIGHT, WIDTH)

  stest = torch.randperm(#test)

   -- Iterate over all images
  K = torch.floor(#test);


  for p=1, K-batchSize,batchSize do


  local err=0;
   -- load one image per person
  for jj=1,batchSize do
    local fn = table.concat({opt.dir,"/",test[stest[p+jj]]});
    testLBLS[jj]=tslbls[stest[p+jj]][1];

   -- print(fn)
   -- make smallest side equal to 256
    im = image.load(fn,3,'float')

    if im:size(3) < im:size(2) then
      im = image.scale(im, 256, 256 * im:size(2) / im:size(3))
   else
      im = image.scale(im, 256 * im:size(3) / im:size(2), 256 )
    end

   local input = im


   local iW = input:size(3)
   local iH = input:size(2)

   -- do random crop
   local oW = WIDTH
   local oH = HEIGHT
   local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
   local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
   local tim = image.crop(input, w1, h1, w1 + oW, h1 + oH)
   assert(tim:size(3) == oW)
   assert(tim:size(2) == oH)

    -- remove mean
    local mean={0.55511024474619, 0.44239095055101,0.33011610164876}
    local stdval={ 0.22127111256688,0.23319352981181, 0.23024111427517}
    for i=1,3 do
      tim[i]:add(-mean[i])
      tim[i]:div(stdval[i])
    end



   if(torch.uniform()>0.5) then
   tim=image.hflip(tim)
   end



    inputs[jj]=tim:clone()

   end
   --take ground truth labels and estimations
  local  lbls=testLBLS:cuda()



  outputs = model:forward(inputs);

   err = criterion:forward(outputs, lbls )


  for i = 1,batchSize do
     confusiontest:add(outputs[i], lbls[i])

  end

  -- disp progress
  xlua.progress(p,K)

  aveloss=aveloss+err
  counter=counter+1;
  -- print("...")
  -- print("AVE LOSS:",aveloss/counter,"LOSS",err)



 end

  -- first print confusion and then add validation values
 print(confusiontest)

 testLogger:add{ ['Loss ='] = aveloss/counter }
 print("Average loss = ",aveloss/counter )
 testLogger:style{['Loss ='] = '-'}
 testLogger:plot()




 testLoggerAcc:add{['% mean class accuracy (test set)'] = confusiontest.totalValid * 100}


 testLoggerAcc:style{['% mean class accuracy (test set)'] = '-'}
 testLoggerAcc:plot()
 confusiontest:zero() -- reset per epoch


end

-- main loop
for epoch=1,Epochs,1 do


   print("Epoch = ",epoch);
   train_epoch()

   test_epoch()


end
