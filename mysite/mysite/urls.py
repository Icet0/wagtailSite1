from django.conf import settings
from django.urls import include, path
from django.contrib import admin

from wagtail.admin import urls as wagtailadmin_urls
from wagtail import urls as wagtail_urls
from wagtail.documents import urls as wagtaildocs_urls
from loadingData.views import load_csv
from context.views import context_view, modal_content_view
from architecture.views import architecture_view
from features.views import features_view
from workflow.views import workflow_view
from dashboard.views import dashboard_view, get_enfants_view, download_file
from visualisation.views import visualisation_view

from search import views as search_views


urlpatterns = [
    path("django-admin/", admin.site.urls),
    path("admin/", include(wagtailadmin_urls)),
    path("documents/", include(wagtaildocs_urls)),
    path("search/", search_views.search, name="search"),
    path('pages/', include(wagtail_urls)),
    
    path('loading/', load_csv, name='load_csv'),
    path('loading/context/', context_view, name='context_view'),
    path('loading/context/modal/', modal_content_view, name='modal_content_view'),




    path('loading/architecture/', architecture_view, name='architecture_view'),
    path('workflow/', workflow_view , name='workflow_view'),
    path('dashboard/', dashboard_view , name='dashboard_view'),

    path('get_enfants_view/', get_enfants_view, name='get_enfants_view'),
    path('download_file/', download_file, name='download_file'),
    
    path('features/', features_view, name='features_view'),
    path('visualisation/', visualisation_view, name='visualisation_view'),
    
    path(r'', include('allauth.urls')),
    

]



if settings.DEBUG:
    from django.conf.urls.static import static
    from django.contrib.staticfiles.urls import staticfiles_urlpatterns

    # Serve static and media files from development server
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns = urlpatterns + [
    # For anything not caught by a more specific rule above, hand over to
    # Wagtail's page serving mechanism. This should be the last pattern in
    # the list:
    path("", include(wagtail_urls)),
    # Alternatively, if you want Wagtail pages to be served from a subpath
    # of your site, rather than the site root:
    #    path("pages/", include(wagtail_urls)),
]
