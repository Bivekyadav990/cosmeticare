from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import post_save
# Create your models here.
class Category(models.Model):
    name = models.CharField(max_length=30, null=True)

    def __str__(self):
        return self.name

class Product(models.Model):
    category = models.ForeignKey(Category, on_delete=models.CASCADE, null=True)
    image = models.FileField(null=True)
    name = models.CharField(max_length=30, null=True)
    price = models.IntegerField(null=True)
    desc = models.TextField(null=True)

    def __str__(self):
        return self.category.name+"--"+self.name

class Status(models.Model):
    name = models.CharField(max_length=20, null=True)

    def __str__(self):
        return self.name

class Profile(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True)
    dob = models.DateField(null=True)
    city = models.CharField(max_length=30, null=True)
    address = models.CharField(max_length=50, null=True)
    contact = models.CharField(max_length=10, null=True)
    image = models.FileField(null=True)

    def __str__(self):
        return self.user.username


class Cart(models.Model):
    profile = models.ForeignKey(Profile,on_delete=models.CASCADE,null=True)
    product = models.ForeignKey(Product, on_delete=models.CASCADE, null=True)
 

    def __str__(self):
        return self.profile.user.username + " . " + self.product.name


class Booking(models.Model):
    status = models.ForeignKey(Status, on_delete=models.CASCADE, null=True)
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, null=True)
    booking_id = models.CharField(max_length=200,null=True)
    quantity = models.CharField(max_length=100,null=True)
    book_date = models.CharField(max_length=30, null=True)
    total = models.IntegerField(null=True)

    def __str__(self):
        return self.book_date+" "+self.profile.user.username



class Send_Feedback(models.Model):
    profile = models.ForeignKey(Profile, on_delete=models.CASCADE, null=True)
    message1 = models.TextField(null=True)
    date = models.CharField(max_length=30, null=True)

    def __str__(self):
        return self.profile.user.username


class Blogpost(models.Model):
    posttitle = models.CharField(max_length=500)
    postdetail = models.TextField()
    postimage = models.FileField()
    postdate = models.DateField()
    def __str__(self):
        return self.posttitle





class TimeStamp(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    modified_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True


class UserPayment(TimeStamp):
    app_user = models.ForeignKey(User, on_delete=models.CASCADE)
    payment_bool = models.BooleanField(default=False)
    stripe_checkout_id = models.CharField(max_length=500)
    amount=models.CharField(max_length=500)
    

@receiver(post_save, sender=User)
def create_user_payment(sender, instance, created, **kwargs):
    if created:
        UserPayment.objects.create(app_user=instance)
