//Çalışma alanının belirlenmesi
var ROI = ee.Geometry.Polygon(
  [ee.Geometry.LinearRing([
    [44.75617064054793,40.66455463804111],
    [44.97040403898543,40.66455463804111],
    [44.97040403898543,40.785801426193736],
    [44.75617064054793,40.785801426193736],
    [44.75617064054793,40.66455463804111]
    ])
  ]
);
Map.addLayer(ROI, {}, 'Çalışma Alanı');

//Bölgenin 2020 tarihli  bulutsuz görüntülerinin çekilemesi ve görselleştirilmesi
var image = ee.ImageCollection('COPERNICUS/S2_SR').filterBounds(ROI).filterMetadata('CLOUD_COVERAGE_ASSESSMENT', 'less_than', 1).filterDate('2020-01-01', '2021-12-31').median().clip(ROI);
Map.addLayer(image, {bands: ['B4', 'B3', 'B2'], min: 0.0, max: 2300.0, gamma: 0.75}, 'Dilican');
Map.centerObject(ROI, 12)

//Yardımcı veri setlerinin hazırlanması
var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
var classification = image.select(['B8', 'B7', 'B4', 'B3', 'B2']).addBands(ndvi);

//Eğitim ve test veri setlerinin çekilmesi
var cayir = ee.FeatureCollection('users/gurolkaba/classification/cayir');
var corakAlan = ee.FeatureCollection('users/gurolkaba/classification/corakAlan');
var orman = ee.FeatureCollection('users/gurolkaba/classification/orman');
var yapi = ee.FeatureCollection('users/gurolkaba/classification/yapi');

//Çekilen verilerin %70 eğitim ve %30 test olacak şekilde ayrılması
var points = cayir.merge(corakAlan).merge(orman).merge(yapi);
var pointsRandom = points.randomColumn();
var training = pointsRandom.filter(ee.Filter.gt('random',0.7));
var validation = pointsRandom.filter(ee.Filter.lte('random',0.3));

//Verilerin eğitilmesi
var trainingData = classification.sampleRegions({
  collection: training,
  properties: ['class_id'],
  scale: 1
  
});

//Sınıflandırma işlemi
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 150
  
});
var classifier = classifier.train({
  features: trainingData,
  classProperty: 'class_id',
  inputProperties: ['B8', 'B4', 'B3', 'B2', 'ndvi']
  
});

//Sınıflandırılmış görüntünün görselleşirilmesi
var classifiedImage = classification.classify(classifier);
Map.addLayer(classifiedImage, {min: 0, max:3, palette: ['4f772d', 'EE9B00', '132a13', '9B2226']}, 'Sınıflandırılmış Görüntü');

//Doğruluk tes matrisi, doğruluk oranları, kappa ve f-score değerlerinin hesaplanması
var testData = classifiedImage.sampleRegions({
  collection: validation,
  properties: ['class_id'],
  scale:1
  
});

var testConfusionMatrix = testData.errorMatrix({
  actual: 'class_id',
  predicted: 'classification'
  
});

print('Doğruluk Matrisi', testConfusionMatrix);
print('Genel Doğruluk',testConfusionMatrix.accuracy());
print('Kullanıcı Doğruluğu', testConfusionMatrix.consumersAccuracy());
print('Üretici Doğruluğu', testConfusionMatrix.producersAccuracy());
print('Kappa', testConfusionMatrix.kappa());
print('F-Fcore', testConfusionMatrix.fscore());